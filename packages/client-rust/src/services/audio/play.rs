use std::process::Stdio;
use std::sync::{Arc, LazyLock};
use tokio::io::AsyncWriteExt;
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;

use crate::base::AppError;

use super::config::{AudioConfig, AUDIO_CONFIG};

pub struct AudioPlayer {
    aplay_thread: Arc<Mutex<Option<Child>>>,
    write_thread: Arc<Mutex<Option<ChildStdin>>>,
    sender: Arc<Mutex<Option<mpsc::Sender<Vec<u8>>>>>,
    player_task: Arc<Mutex<Option<JoinHandle<()>>>>,
}

static INSTANCE: LazyLock<AudioPlayer> = LazyLock::new(AudioPlayer::new);

impl AudioPlayer {
    fn new() -> Self {
        Self {
            aplay_thread: Arc::new(Mutex::new(None)),
            write_thread: Arc::new(Mutex::new(None)),
            sender: Arc::new(Mutex::new(None)),
            player_task: Arc::new(Mutex::new(None)),
        }
    }

    pub fn instance() -> &'static Self {
        &INSTANCE
    }

    pub async fn stop(&self) -> Result<(), AppError> {
        let mut sender_guard = self.sender.lock().await;
        if let Some(sender) = sender_guard.take() {
            drop(sender);
        }

        if let Some(task) = self.player_task.lock().await.take() {
            task.abort();
        }

        if let Some(mut write_thread) = self.write_thread.lock().await.take() {
            let _ = write_thread.shutdown().await;
            drop(write_thread);
        }

        if let Some(mut aplay_thread) = self.aplay_thread.lock().await.take() {
            let _ = aplay_thread.kill().await;
        }

        Ok(())
    }

    pub async fn start(&self, config: Option<AudioConfig>) -> Result<(), AppError> {
        let is_started = self.sender.lock().await.is_some();
        if is_started {
            self.stop().await?;
        }

        let config = config.unwrap_or_else(|| (*AUDIO_CONFIG).clone());

        let mut aplay_thread = Command::new("aplay")
            .args([
                "--quiet",
                "-t",
                "raw",
                "-f",
                &format!("S{}_LE", config.bits_per_sample),
                "-r",
                &config.sample_rate.to_string(),
                "-c",
                &config.channels.to_string(),
                "--buffer-size",
                &config.buffer_size.to_string(),
                "--period-size",
                &config.period_size.to_string(),
                "-",
            ])
            .stdin(Stdio::piped())
            .spawn()?;

        let stdin = aplay_thread.stdin.take().unwrap();
        self.aplay_thread.lock().await.replace(aplay_thread);
        self.write_thread.lock().await.replace(stdin);

        let (tx, mut rx) = mpsc::channel::<Vec<u8>>(100);

        let write_thread_clone = self.write_thread.clone();
        let player_task = tokio::spawn(async move {
            while let Some(bytes) = rx.recv().await {
                let mut write_guard = write_thread_clone.lock().await;
                if let Some(write_thread) = write_guard.as_mut() {
                    let _ = write_thread.write_all(&bytes).await;
                } else {
                    break;
                }
            }
        });

        self.player_task.lock().await.replace(player_task);
        self.sender.lock().await.replace(tx);

        Ok(())
    }

    pub async fn play(&self, bytes: Vec<u8>) -> Result<(), AppError> {
        let sender_guard = self.sender.lock().await;
        if let Some(sender) = sender_guard.as_ref() {
            sender.send(bytes).await?;
        }

        Ok(())
    }
}
