use std::{
    collections::HashMap,
    future::Future,
    sync::{Arc, LazyLock},
};

use tokio::{sync::Mutex, task::JoinHandle};

/// 批量管理异步任务，在合适的时机终止
pub struct TaskManager {
    tasks: Arc<Mutex<HashMap<String, Vec<JoinHandle<()>>>>>,
}

static INSTANCE: LazyLock<TaskManager> = LazyLock::new(TaskManager::new);

impl TaskManager {
    fn new() -> Self {
        Self {
            tasks: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn instance() -> &'static Self {
        &INSTANCE
    }

    pub async fn add(&self, tag: &str, handle: JoinHandle<()>) {
        let mut tasks = self.tasks.lock().await;
        let handles = tasks.entry(tag.to_string()).or_insert_with(Vec::new);
        handles.retain(|h| !h.is_finished());
        handles.push(handle);
    }

    pub async fn dispose(&self, tag: &str) {
        let mut tasks = self.tasks.lock().await;
        if let Some(handles) = tasks.remove(tag) {
            for handle in handles {
                handle.abort();
            }
        }
    }

    pub async fn run_async<F>(future: F)
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let task = tokio::spawn(async move {
            let _ = future.await;
        });

        TaskManager::instance().add("async", task).await;
    }

    pub async fn dispose_async() {
        TaskManager::instance().dispose("async").await;
    }
}
