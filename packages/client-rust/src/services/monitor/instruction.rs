use std::future::Future;

use serde::{Deserialize, Serialize};

use crate::base::AppError;

use super::file::{FileMonitor, FileMonitorEvent};

static INSTRUCTION_FILE_PATH: &str = "/tmp/mico_aivs_lab/instruction.log";

pub struct InstructionMonitor {
    file_monitor: FileMonitor,
}

impl Default for InstructionMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl InstructionMonitor {
    pub fn new() -> Self {
        Self {
            file_monitor: FileMonitor::new(),
        }
    }

    pub async fn start<F, Fut>(&mut self, on_update: F)
    where
        F: Fn(FileMonitorEvent) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<(), AppError>> + Send + 'static,
    {
        self.file_monitor
            .start(INSTRUCTION_FILE_PATH, on_update)
            .await;
    }

    pub async fn stop(&mut self) {
        self.file_monitor.stop().await;
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Header {
    pub dialog_id: String,
    pub id: String,
    pub name: String,
    pub namespace: String,
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct RecognizeResult {
    #[serde(default)]
    pub confidence: f64,
    #[serde(default)]
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub asr_binary_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub begin_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_nlp_request: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_stop: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin_text: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Payload {
    RecognizeResultPayload {
        is_final: bool,
        is_vad_begin: bool,
        results: Vec<RecognizeResult>,
    },
    StopCapturePayload {
        stop_time: u64,
    },
    SpeakPayload {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        emotion: Option<Emotion>,
    },
    PlayPayload {
        audio_items: Vec<AudioItem>,
        audio_type: String,
        loadmore_token: String,
        needs_loadmore: bool,
        origin_id: String,
        play_behavior: String,
    },
    SetPropertyPayload {
        name: String,
        value: String,
    },
    InstructionControlPayload {
        behavior: String,
    },
    EmptyPayload {},
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Emotion {
    pub category: String,
    pub level: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AudioItem {
    pub item_id: ItemId,
    pub log: Log,
    pub stream: Stream,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ItemId {
    pub audio_id: String,
    pub cp: Cp,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Cp {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Log {
    pub eid: String,
    pub refer: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Stream {
    pub authentication: bool,
    pub duration_in_ms: u64,
    pub offset_in_ms: u64,
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogMessage {
    pub header: Header,
    pub payload: Payload,
}
