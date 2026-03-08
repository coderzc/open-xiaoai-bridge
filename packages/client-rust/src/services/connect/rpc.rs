use futures::future::BoxFuture;
use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, LazyLock};
use tokio::sync::{oneshot, Mutex, RwLock};
use tokio::time::{timeout, Duration};
use uuid::Uuid;

use crate::base::AppError;

use super::data::{Request, Response};

type SendRequestFn = Arc<dyn Fn(Request) -> BoxFuture<'static, Result<(), AppError>> + Send + Sync>;

type RequestHandler =
    Arc<dyn Fn(Request) -> BoxFuture<'static, Result<Response, AppError>> + Send + Sync>;

static INSTANCE: LazyLock<RPC> = LazyLock::new(RPC::new);

pub struct RPC {
    send_request: Arc<RwLock<Option<SendRequestFn>>>,
    request_handlers: Arc<RwLock<HashMap<String, RequestHandler>>>,
    pending_requests: Arc<Mutex<HashMap<String, oneshot::Sender<Response>>>>,
}

impl RPC {
    fn new() -> Self {
        Self {
            send_request: Arc::new(RwLock::new(None)),
            request_handlers: Arc::new(RwLock::new(HashMap::new())),
            pending_requests: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn instance() -> &'static Self {
        &INSTANCE
    }

    pub async fn init<F, Fut>(&self, send_request: F)
    where
        F: Fn(Request) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), AppError>> + Send + 'static,
    {
        let mut lock = self.send_request.write().await;
        *lock = Some(Arc::new(move |msg| Box::pin(send_request(msg))));
    }

    pub async fn dispose(&self) {
        {
            let mut lock = self.send_request.write().await;
            *lock = None;
        }
        {
            let mut lock = self.request_handlers.write().await;
            lock.clear();
        }
        {
            let mut lock = self.pending_requests.lock().await;
            lock.clear();
        }
    }

    /// 添加 local 方法
    pub async fn add_command<F, Fut>(&self, command: &str, handler: F)
    where
        F: Fn(Request) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Response, AppError>> + Send + 'static,
    {
        let mut handlers = self.request_handlers.write().await;
        handlers.insert(
            command.to_string(),
            Arc::new(move |request| Box::pin(handler(request))),
        );
    }

    /// local 收到 remote 调用
    pub async fn on_request(&self, request: Request) -> Result<Response, AppError> {
        let handler = self.get_request_handler(&request.command).await;
        match handler {
            Some(handler) => handler(request).await,
            None => Err("command not found".into()),
        }
    }

    /// local 收到 remote 响应
    pub async fn on_response(&self, response: Response) {
        let tx = {
            let mut pending = self.pending_requests.lock().await;
            pending.remove(&response.id)
        };

        if let Some(tx) = tx {
            let _ = tx.send(response);
        }
    }

    /// 调用 remote 方法
    pub async fn call_remote(
        &self,
        command: &str,
        payload: Option<serde_json::Value>,
        timeout_millis: Option<u64>,
    ) -> Result<Response, AppError> {
        let send_request = self.get_send_request().await;
        let Some(send_request) = send_request else {
            return Err("send_request is not initialized".into());
        };

        let uid = Uuid::new_v4().to_string();
        let (tx, rx) = oneshot::channel();

        let request = Request {
            id: uid.clone(),
            command: command.to_string(),
            payload,
        };

        send_request(request).await?;

        {
            let mut pending = self.pending_requests.lock().await;
            pending.insert(uid.clone(), tx);
        }

        let timeout_duration = Duration::from_millis(timeout_millis.unwrap_or(10 * 1000));
        match timeout(timeout_duration, rx).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(_)) => {
                let mut pending = self.pending_requests.lock().await;
                pending.remove(&uid);
                Err("response channel closed".into())
            }
            Err(_) => {
                let mut pending = self.pending_requests.lock().await;
                pending.remove(&uid);
                Err("request timeout".into())
            }
        }
    }

    async fn get_send_request(&self) -> Option<SendRequestFn> {
        let send_request = self.send_request.read().await;
        send_request.clone()
    }

    async fn get_request_handler(&self, event: &str) -> Option<RequestHandler> {
        let handlers = self.request_handlers.read().await;
        handlers.get(event).cloned()
    }
}
