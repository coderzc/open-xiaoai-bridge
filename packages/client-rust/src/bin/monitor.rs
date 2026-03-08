use std::path::Path;

use open_xiaoai::{
    services::monitor::kws::{KwsMonitor, KwsMonitorEvent, KWS_FILE_PATH},
    utils::{rand::pick_one, shell::run_shell},
};

static KWS_REPLY_FILE_PATH: &str = "/data/open-xiaoai/kws/reply.txt";

async fn on_started() {
    let welcome = std::env::args()
        .nth(1)
        .unwrap_or("自定义唤醒词已开启".to_string());
    let _ = run_shell(&format!("/usr/sbin/tts_play.sh '{}'", welcome)).await;
}

async fn on_keyword(_keyword: String) {
    println!("🔥 唤醒词: {}", _keyword);

    let mut wakeup_sounds = vec![
        "file:///usr/share/sound-vendor/AiNiRobot/wakeup_ei_01.wav".to_string(),
        "file:///usr/share/sound-vendor/AiNiRobot/wakeup_zai_01.wav".to_string(),
    ];

    if Path::new(KWS_REPLY_FILE_PATH).exists() {
        // 播放自定义唤醒提示音
        let content = std::fs::read_to_string(KWS_REPLY_FILE_PATH).unwrap();
        let replies = content
            .split("\n")
            .filter(|line| !line.trim().is_empty())
            .collect::<Vec<&str>>();
        if !replies.is_empty() {
            wakeup_sounds.clear();
            wakeup_sounds.extend(replies.iter().map(|s| s.to_string()));
        }
    }

    let reply = pick_one(&wakeup_sounds);
    let script = if reply.contains("://") {
        format!("miplayer -f '{}'", reply)
    } else {
        format!("/usr/sbin/tts_play.sh '{}'", reply)
    };
    let _ = run_shell(&script).await;

    // 唤醒
    let _ = run_shell(r#"ubus call pnshelper event_notify '{"src":1,"event":0}'"#).await;
}

#[tokio::main]
async fn main() {
    if Path::new(KWS_FILE_PATH).exists() {
        on_started().await;
    }

    KwsMonitor::new()
        .start(|event| async move {
            match event {
                KwsMonitorEvent::Started => on_started().await,
                KwsMonitorEvent::Keyword(keyword) => on_keyword(keyword).await,
            }
            Ok(())
        })
        .await;

    loop {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}
