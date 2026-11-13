use crate::python::PythonManager;
use open_xiaoai::base::{AppError, VERSION};
use open_xiaoai::services::audio::config::AudioConfig;
use open_xiaoai::services::connect::data::{Event, Request, Response, Stream};
use open_xiaoai::services::connect::handler::MessageHandler;
use open_xiaoai::services::connect::message::{MessageManager, WsStream};
use open_xiaoai::services::connect::rpc::RPC;
use open_xiaoai::services::speaker::SpeakerManager;
use open_xiaoai::utils::task::TaskManager;
use pyo3::types::PyBytes;
use pyo3::types::PyString;
use pyo3::Python;
use serde_json::json;
use std::env;
use std::sync::LazyLock;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{oneshot, Mutex};
use tokio::task::JoinHandle;
use tokio::time::{timeout, Duration};
use tokio_tungstenite::{accept_async, accept_hdr_async};

pub struct AppServer;

struct ActiveConnection {
    addr: std::net::SocketAddr,
    shutdown_tx: Option<oneshot::Sender<()>>,
    task: JoinHandle<()>,
}

impl ActiveConnection {
    async fn stop(mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }

        if timeout(Duration::from_secs(2), &mut self.task)
            .await
            .is_err()
        {
            self.task.abort();
            let _ = self.task.await;
        }
    }
}

static ACTIVE_CONNECTION: LazyLock<Mutex<Option<ActiveConnection>>> =
    LazyLock::new(|| Mutex::new(None));

/// Check if audio input is enabled via environment variable.
/// Supports: "true"/"false", "1"/"0", "yes"/"no", etc.
/// Defaults to true if not set or invalid.
fn is_audio_input_enabled() -> bool {
    match env::var("AUDIO_INPUT_ENABLE") {
        Ok(val) => {
            let val = val.trim().to_lowercase();
            matches!(val.as_str(), "true" | "1" | "yes" | "on")
        }
        Err(_) => true,
    }
}

async fn test() -> Result<(), AppError> {
    SpeakerManager::play_text("已连接").await?;

    // Only start recording if audio input is enabled
    if is_audio_input_enabled() {
        let _ = RPC::instance()
            .call_remote(
                "start_recording",
                Some(json!(AudioConfig {
                    pcm: "noop".into(),
                    channels: 1,
                    bits_per_sample: 16,
                    sample_rate: 16000,
                    period_size: 1440 / 4,
                    buffer_size: 1440,
                })),
                None,
            )
            .await;
    }

    // aplay is started lazily by ensure_player_ready() on first audio send,
    // avoiding empty-buffer underruns from idling aplay processes.

    Ok(())
}

impl AppServer {
    pub async fn connect(stream: TcpStream) -> Result<WsStream, AppError> {
        let expected_token = std::env::var("OPEN_XIAOAI_TOKEN").unwrap_or_default();
        if !expected_token.is_empty() {
            use tokio_tungstenite::tungstenite::handshake::server::{
                ErrorResponse, Request, Response,
            };
            let ws_stream = accept_hdr_async(stream, move |req: &Request, response: Response| {
                let auth = req
                    .headers()
                    .get("Authorization")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("");
                if auth != format!("Bearer {}", expected_token) {
                    let error: ErrorResponse =
                        tokio_tungstenite::tungstenite::http::Response::builder()
                            .status(401)
                            .body(Some("Unauthorized".to_string()))
                            .unwrap();
                    return Err(error);
                }
                Ok(response)
            })
            .await?;
            Ok(WsStream::Server(ws_stream))
        } else {
            let ws_stream = accept_async(stream).await?;
            Ok(WsStream::Server(ws_stream))
        }
    }

    pub async fn run() {
        let addr = "0.0.0.0:4399";
        let listener = TcpListener::bind(&addr)
            .await
            .expect(format!("[AppServer] ❌ 绑定地址失败: {}", &addr).as_str());
        crate::pylog!("[AppServer] ✅ 已启动: {:?}", addr);
        while let Ok((stream, addr)) = listener.accept().await {
            let previous = {
                let mut active_connection = ACTIVE_CONNECTION.lock().await;
                active_connection.take()
            };
            if let Some(previous) = previous {
                crate::pylog!("[AppServer] ⚠️ 发现新连接，关闭旧连接: {}", previous.addr);
                previous.stop().await;
            }

            let (shutdown_tx, shutdown_rx) = oneshot::channel();
            let task = tokio::spawn(async move {
                AppServer::handle_connection(stream, addr, shutdown_rx).await;
            });
            let mut active_connection = ACTIVE_CONNECTION.lock().await;
            *active_connection = Some(ActiveConnection {
                addr,
                shutdown_tx: Some(shutdown_tx),
                task,
            });
        }
    }

    async fn handle_connection(
        stream: TcpStream,
        addr: std::net::SocketAddr,
        mut shutdown_rx: oneshot::Receiver<()>,
    ) {
        let ws_stream = match AppServer::connect(stream).await {
            Ok(ws_stream) => ws_stream,
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("401") || msg.contains("Unauthorized") {
                    crate::pylog!("[AppServer] ❌ 鉴权失败: {}", addr);
                } else {
                    crate::pylog!("[AppServer] ❌ 连接异常: {} ({})", addr, msg);
                }
                return;
            }
        };
        crate::pylog!("[AppServer] ✅ 已连接: {:?}", addr);
        AppServer::init(ws_stream).await;
        tokio::select! {
            result = MessageManager::instance().process_messages() => {
                if let Err(e) = result {
                    crate::pylog!("[AppServer] ❌ 消息处理异常: {}", e);
                }
            }
            _ = &mut shutdown_rx => {
                crate::pylog!("[AppServer] ⚠️ 连接被新会话抢占: {}", addr);
            }
        }
        AppServer::dispose().await;
        crate::pylog!("[AppServer] ❌ 已断开连接");
    }

    async fn init(ws_stream: WsStream) {
        MessageManager::instance().init(ws_stream).await;
        MessageHandler::<Event>::instance()
            .set_handler(on_event)
            .await;
        MessageHandler::<Stream>::instance()
            .set_handler(on_stream)
            .await;

        let rpc = RPC::instance();
        rpc.add_command("get_version", get_version).await;

        let test = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let _ = test().await;
        });
        TaskManager::instance().add("test", test).await;
    }

    async fn dispose() {
        MessageManager::instance().dispose().await;
        TaskManager::instance().dispose("test").await;
    }
}

async fn get_version(_: Request) -> Result<Response, AppError> {
    let data = json!(VERSION.to_string());
    Ok(Response::from_data(data))
}

async fn on_stream(stream: Stream) -> Result<(), AppError> {
    let Stream { tag, bytes, .. } = stream;
    match tag.as_str() {
        "record" => {
            let data = Python::with_gil(|py| PyBytes::new(py, &bytes).into());
            PythonManager::instance().call_fn("on_input_data", Some(data))?;
        }
        _ => {}
    }
    Ok(())
}

async fn on_event(event: Event) -> Result<(), AppError> {
    let event_json = serde_json::to_string(&event)?;
    let data = Python::with_gil(|py| PyString::new(py, &event_json).into());
    PythonManager::instance().call_fn("on_event", Some(data))?;
    Ok(())
}
