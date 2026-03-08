use serde_json::json;

use crate::utils::shell::CommandResult;
use crate::{base::AppError, services::connect::rpc::RPC};

pub struct SpeakerManager;

impl SpeakerManager {
    /// 获取启动分区
    pub async fn get_boot() -> Result<String, AppError> {
        const COMMAND: &str = r#"
            echo $(fw_env -g boot_part)
        "#;
        let res = SpeakerManager::run_shell(COMMAND).await?;
        Ok(res.stdout.trim().to_string())
    }

    /// 设置启动分区
    pub async fn set_boot(boot_part: &str) -> Result<bool, AppError> {
        const COMMAND: &str = r#"
            fw_env -s boot_part %s >/dev/null 2>&1 && echo $(fw_env -g boot_part)
        "#;
        let script = COMMAND.replace("%s", boot_part);
        let res = SpeakerManager::run_shell(&script).await?;
        Ok(res.stdout.contains(boot_part))
    }

    /// 获取设备型号
    pub async fn get_device_model() -> Result<String, AppError> {
        const COMMAND: &str = r#"
            echo $(micocfg_model)
        "#;
        let res = SpeakerManager::run_shell(COMMAND).await?;
        Ok(res.stdout.trim().to_string())
    }

    /// 获取设备序列号
    pub async fn get_device_sn() -> Result<String, AppError> {
        const COMMAND: &str = r#"
            echo $(micocfg_sn)
        "#;
        let res = SpeakerManager::run_shell(COMMAND).await?;
        Ok(res.stdout.trim().to_string())
    }

    /// 获取播放状态
    pub async fn get_play_status() -> Result<String, AppError> {
        const COMMAND: &str = r#"
            mphelper mute_stat
        "#;
        let res = SpeakerManager::run_shell(COMMAND).await?;
        let status = if res.stdout.contains("1") {
            "playing"
        } else if res.stdout.contains("2") {
            "paused"
        } else {
            "idle"
        };
        Ok(status.to_string())
    }

    /// 播放
    pub async fn play() -> Result<bool, AppError> {
        const COMMAND: &str = r#"
            mphelper play
        "#;
        let res = SpeakerManager::run_shell(COMMAND).await?;
        Ok(res.stdout.contains("\"code\": 0"))
    }

    /// 暂停
    pub async fn pause() -> Result<bool, AppError> {
        const COMMAND: &str = r#"
            mphelper pause
        "#;
        let res = SpeakerManager::run_shell(COMMAND).await?;
        Ok(res.stdout.contains("\"code\": 0"))
    }

    /// TTS
    pub async fn play_text(text: &str) -> Result<bool, AppError> {
        const COMMAND: &str = r#"
            /usr/sbin/tts_play.sh '%s'
        "#;
        let script = COMMAND.replace("%s", text);
        let res = SpeakerManager::run_shell(&script).await?;
        Ok(res.stdout.contains("\"code\": 0"))
    }

    /// 播放音频
    pub async fn play_url(url: &str) -> Result<bool, AppError> {
        const COMMAND: &str = r#"
            ubus call mediaplayer player_play_url '{"url":"%s","type": 1}'
        "#;
        let script = COMMAND.replace("%s", url);
        let res = SpeakerManager::run_shell(&script).await?;
        Ok(res.stdout.contains("\"code\": 0"))
    }

    /// 获取麦克风状态
    pub async fn get_mic_status() -> Result<String, AppError> {
        const COMMAND: &str = r#"
            [ ! -f /tmp/mipns/mute ] && echo on || echo off
        "#;
        let res = SpeakerManager::run_shell(COMMAND).await?;
        let status = if res.stdout.contains("on") {
            "on"
        } else {
            "off"
        };
        Ok(status.to_string())
    }

    /// 打开麦克风
    pub async fn mic_on() -> Result<bool, AppError> {
        const COMMAND: &str = r#"
            ubus -t1 -S call pnshelper event_notify '{"src":3, "event":7}' 2>&1
        "#;
        let res = SpeakerManager::run_shell(COMMAND).await?;
        Ok(res.stdout.contains("\"code\":0"))
    }

    /// 关闭麦克风
    pub async fn mic_off() -> Result<bool, AppError> {
        const COMMAND: &str = r#"
            ubus -t1 -S call pnshelper event_notify '{"src":3, "event":8}' 2>&1
        "#;
        let res = SpeakerManager::run_shell(COMMAND).await?;
        Ok(res.stdout.contains("\"code\":0"))
    }

    /// 执行命令
    pub async fn ask_xiaoai(text: &str) -> Result<bool, AppError> {
        const COMMAND: &str = r#"
            ubus call mibrain ai_service '{"tts":1,"nlp":1,"nlp_text":"%s"}'
        "#;
        let script = COMMAND.replace("%s", text);
        let res = SpeakerManager::run_shell(&script).await?;
        Ok(res.stdout.contains("\"code\": 0"))
    }

    /// 中断运行
    pub async fn abort_xiaoai() -> Result<bool, AppError> {
        const COMMAND: &str = r#"
            /etc/init.d/mico_aivs_lab restart >/dev/null 2>&1
        "#;
        let res = SpeakerManager::run_shell(COMMAND).await?;
        Ok(res.exit_code == 0)
    }

    /// 唤醒
    pub async fn wake_up(flag: bool) -> Result<bool, AppError> {
        let command = if flag {
            r#"
                ubus call pnshelper event_notify '{"src":1,"event":0}'
            "#
        } else {
            r#"
                ubus call pnshelper event_notify '{"src":3, "event":7}'
                sleep 0.1
                ubus call pnshelper event_notify '{"src":3, "event":8}'
            "#
        };
        let res = SpeakerManager::run_shell(command).await?;
        Ok(res.stdout.contains("\"code\": 0"))
    }

    async fn run_shell(script: &str) -> Result<CommandResult, AppError> {
        let res = RPC::instance()
            .call_remote("run_shell", Some(json!(script)), None)
            .await?;
        Ok(serde_json::from_value::<CommandResult>(res.data.unwrap()).unwrap())
    }
}
