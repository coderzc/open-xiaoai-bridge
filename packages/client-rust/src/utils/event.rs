use futures::future::BoxFuture;
use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, LazyLock},
};
use tokio::sync::Mutex;

use crate::{base::AppError, services::connect::data::Event};

use super::task::TaskManager;

type EventCallback = Arc<dyn Fn(Event) -> BoxFuture<'static, Result<(), AppError>> + Send + Sync>;

pub struct EventBus {
    subscribers: Arc<Mutex<HashMap<String, Vec<EventCallback>>>>,
}

static INSTANCE: LazyLock<EventBus> = LazyLock::new(EventBus::new);

impl EventBus {
    fn new() -> Self {
        EventBus {
            subscribers: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn instance() -> &'static Self {
        &INSTANCE
    }

    pub async fn subscribe<F, Fut>(&self, event: &str, callback: F)
    where
        F: Fn(Event) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), AppError>> + Send + 'static,
    {
        let mut subscribers = self.subscribers.lock().await;
        subscribers
            .entry(event.to_string())
            .or_insert_with(Vec::new)
            .push(Arc::new(move |event| Box::pin(callback(event))));
    }

    pub async fn unsubscribe(&self, event: &str) {
        self.subscribers.lock().await.remove(event);
        let tag = format!("EventBus-{}", event);
        TaskManager::instance().dispose(&tag).await;
    }

    pub async fn publish(&self, event: Event) {
        let Some(callbacks) = self.get_callbacks(&event.event).await else {
            return;
        };
        for callback in callbacks {
            let event = event.clone();
            let _ = callback(event).await;
        }
    }

    pub async fn publish_async(&self, event: Event) {
        let Some(callbacks) = self.get_callbacks(&event.event).await else {
            return;
        };
        let tag = format!("EventBus-{}", event.event);
        for callback in callbacks {
            let event = event.clone();
            let task = tokio::spawn(async move {
                let _ = callback(event).await;
            });
            TaskManager::instance().add(&tag, task).await;
        }
    }

    async fn get_callbacks(&self, event: &str) -> Option<Vec<EventCallback>> {
        let subscribers = self.subscribers.lock().await;
        subscribers.get(event).cloned()
    }
}
