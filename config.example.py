import uuid


# 每次唤醒生成独立 Session，对话互不干扰
# 适合以下场景：
#   - "提问 → 回答"式交互，不需要 Agent 记住上下文
#   - 长期使用同一 Session 导致 Agent 上下文窗口堆积过长，影响响应质量和速度
# def new_session_key():
#     return f"agent:main:session-{uuid.uuid4().hex[:8]}"


async def before_wakeup(speaker, text, source, app):
    if source == "kws":
        if "螃蟹" in text:
            await speaker.play(text="螃蟹来了")
            return "openclaw"

        if "小智" in text:
            await speaker.play(text="小智来了")
            return "xiaozhi"

        return None

    if source == "xiaoai":
        if text == "召唤螃蟹":
            await speaker.abort_xiaoai()
            return "openclaw"

        if text == "召唤小智":
            await speaker.abort_xiaoai()
            return "xiaozhi"

        if "请" in text:
            await speaker.abort_xiaoai()
            await app.send_to_openclaw_and_play_reply(text.replace("请", ""))
            return None

        if "告诉螃蟹" in text:
            await speaker.abort_xiaoai()
            await app.send_to_openclaw_and_play_reply(text.replace("告诉螃蟹", ""))
            return None


async def after_wakeup(speaker, source=None, session_key=None):
    if source in {"openclaw", "xiaozhi"}:
        await speaker.play(text="再见")


APP_CONFIG = {
    "wakeup": {
        "provider": "doubao",
        "keywords": [
            "你好小智",
            "小智小智",
            "hi open claw",
            "你好螃蟹",
            "螃蟹你好",
            "螃蟹",
            "螃蟹螃蟹",
        ],
        "doubao": {
            "mode": "utterance_end",
            "asr_provider": "openai_compatible",
            "window_ms": 3500,
            "check_interval_ms": 1800,
            "min_gap_ms": 1500,
            "min_audio_bytes": 20000,
            "max_audio_ms": 2000,
            "max_audio_bytes": 64000,
            "utterance_end_silence_frames": 12,
            "utterance_max_ms": 5000,
            "input_gain": 8.0,
        },
        "timeout": 20,
        "before_wakeup": before_wakeup,
        "after_wakeup": after_wakeup,
    },
    "kws": {
        "keywords_score": 0.1,
        "keywords_threshold": 0.1,
        "min_silence_duration": 480,
    },
    "vad": {
        "threshold": 0.10,
        "min_speech_duration": 250,
        "min_silence_duration": 1000,
    },
    "asr": {
        "provider": "openai_compatible",
        "fallback_provider": None,
        "contexts": {
            "conversation": {
                "provider": "openai_compatible",
                "fallback_provider": None,
                "max_speech_ms": 30000,
                "max_audio_bytes": 960000,
            },
            "wakeup": {
                "provider": "openai_compatible",
                "fallback_provider": None,
            },
        },
        "model": "bigmodel",
        "timeout": 20,
        "replacements": {},
        "openai_compatible": {
            "base_url": "http://127.0.0.1:8787",
            "api_key": "YOUR_OPENAI_COMPATIBLE_API_KEY",
        },
    },
    "audio_debug": {
        "enabled": False,
        "contexts": {
            "wakeup": {
                "save_wav": False,
                "dir": "/app/core/debug_wakeup_audio",
                "max_files": 10,
                "input_gain": 8.0,
            },
            "conversation": {
                "save_wav": False,
                "dir": "/app/core/debug_conversation_audio",
                "max_files": 10,
                "input_gain": 8.0,
            },
        },
    },
    "xiaozhi": {
        "OTA_URL": "http://127.0.0.1:8003/xiaozhi/ota/",
        "WEBSOCKET_URL": "ws://127.0.0.1:8000/xiaozhi/v1/",
        "WEBSOCKET_ACCESS_TOKEN": "",
        "DEVICE_ID": "",
        "VERIFICATION_CODE": "",
    },
    "xiaoai": {
        "continuous_conversation_mode": True,
        "exit_command_keywords": ["停止", "退下", "退出", "下去吧"],
        "max_listening_retries": 2,
        "exit_prompt": "再见，主人",
        "continuous_conversation_keywords": ["开启连续对话", "启动连续对话", "我想跟你聊天"],
    },
    "tts": {
        "doubao": {
            "app_id": "YOUR_DOUBAO_APP_ID",
            "access_key": "YOUR_DOUBAO_ACCESS_KEY",
            "default_speaker": "saturn_zh_female_keainvsheng_tob",
            "audio_format": "pcm",
            "stream": True,
        }
    },
    "openclaw": {
        "url": "ws://127.0.0.1:18789",
        "token": "YOUR_OPENCLAW_TOKEN",
        "input_mode": "local_asr",
        "session_key": "agent:speaker:open-xiaoai-bridge",
        "identity_path": "/app/openclaw/identity/device.json",
        "tts_speed": 1.0,
        "tts_speaker": "xiaoai",
        "agent_tts_speakers": {},
        "response_timeout": 120,
        "exit_keywords": ["退出", "停止", "再见"],
        "rule_prompt": "在客厅，请返回300字以内的纯文字信息",
        "rule_prompt_for_skill": "注意：这条消息是主人通过小爱音箱发送的，他看不到你回复的文字，调用 `xiaoai-tts` skill 播报出来。字数控制在300字以内",
    },
}
