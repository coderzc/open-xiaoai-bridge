use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::base::AppError;

use super::file::{FileMonitor, FileMonitorEvent};

pub static KWS_FILE_PATH: &str = "/tmp/open-xiaoai/kws.log";

#[derive(Debug, Serialize, Deserialize)]
pub enum KwsMonitorEvent {
    Started,
    Keyword(String),
}

pub struct KwsMonitor {
    file_monitor: FileMonitor,
}

impl Default for KwsMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl KwsMonitor {
    pub fn new() -> Self {
        Self {
            file_monitor: FileMonitor::new(),
        }
    }

    pub async fn start<F, Fut>(&mut self, on_update: F)
    where
        F: Fn(KwsMonitorEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), AppError>> + Send + 'static,
    {
        let on_update = Arc::new(on_update);
        let last_ts_store = Arc::new(AtomicU64::new(0));

        self.file_monitor
            .start(KWS_FILE_PATH, move |event| {
                let on_update = Arc::clone(&on_update);
                let last_ts_store = Arc::clone(&last_ts_store);

                async move {
                    if let FileMonitorEvent::NewLine(content) = event {
                        let data = content.split('@').collect::<Vec<&str>>();
                        let timestamp = data[0].parse::<u64>().unwrap();
                        let keyword = data[1].to_string();
                        let last_timestamp = last_ts_store.load(Ordering::Relaxed);
                        if timestamp != last_timestamp {
                            last_ts_store.store(timestamp, Ordering::Relaxed);
                            let kws_event = if keyword == "__STARTED__" {
                                KwsMonitorEvent::Started
                            } else {
                                KwsMonitorEvent::Keyword(keyword)
                            };
                            let _ = on_update(kws_event).await;
                        }
                    }
                    Ok(())
                }
            })
            .await;
    }

    pub async fn stop(&mut self) {
        self.file_monitor.stop().await;
    }
}
