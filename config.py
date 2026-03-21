import asyncio
import socket
import subprocess
import sys
import time

import requests


async def before_wakeup(speaker, text, source, app):
    """
    处理收到的用户消息，并决定是否唤醒小智 AI

    - source: 唤醒来源
        - 'kws': 关键字唤醒
        - 'xiaoai': 小爱同学收到用户指令

    返回值:
        - "xiaozhi": 走小智流程
        - "openclaw": 走 OpenClaw 连续对话流程
        - None: 不处理（用户可在此自行调用 app.send_to_openclaw 等方法）
    """
    if source == "kws":
        # Check if the keyword matches an OpenClaw wake word
        if "龙虾" in text:
            await speaker.play(text="龙虾来了")
            return "openclaw"

        if "小智" in text:
            await speaker.play(text="小智来了")
            return "xiaozhi"
        return None

    if source == "xiaoai":
        if text == "召唤龙虾":
            await speaker.abort_xiaoai()
            return "openclaw"  # OpenClaw 连续对话

        if text == "召唤小智":
            await speaker.abort_xiaoai()
            return "xiaozhi"  # 小智 AI

        if "让龙虾" in text:
            await speaker.abort_xiaoai()
            # 单次代理：发送给 OpenClaw 并自动 TTS 播报回复
            await app.send_to_openclaw_and_play_reply(text.replace("让龙虾", ""))
            return None  # 框架不做额外处理

        if "告诉龙虾" in text:
            await speaker.abort_xiaoai()
            # 只发送，不播报（由 Agent 自行调用 xiaoai-tts skill 播报）
            await app.send_to_openclaw(text.replace("告诉龙虾", ""))
            return None


async def after_wakeup(speaker, source="xiaozhi"):
    """
    退出唤醒状态

    - source: 退出来源
        - 'xiaozhi': 小智对话超时退出
        - 'openclaw': OpenClaw 连续对话退出
    """
    if source == "openclaw":
        await speaker.play(text="龙虾，再见")
    if source == "xiaozhi":
        await speaker.play(text="小智，再见")

APP_CONFIG = {
    "wakeup": {
        # 自定义唤醒词列表（英文字母要全小写）
        "keywords": [
            "你好小智",
            "小智小智",
            "hi open claw",
            "你好龙虾",
            "龙虾你好",
        ],
        # 静音多久后自动退出唤醒（秒）
        "timeout": 20,
        # 语音识别结果回调
        "before_wakeup": before_wakeup,
        # 退出唤醒时的提示语（设置为空可关闭）
        "after_wakeup": after_wakeup,
    },
    "kws": {
        # 唤醒词置信度加成（越高越难误触发，越低越灵敏）
        "keywords_score": 2.0,
        # 唤醒词检测阈值（越低越灵敏，越高越难触发）
        "keywords_threshold": 0.2,
        # 唤醒词检测时的最小静默时长（ms），静默超过该时长则判定为说完
        "min_silence_duration": 480,
    },
    "vad": {
        # 语音检测阈值（0-1，越小越灵敏）
        "threshold": 0.10,
        # 最小语音时长（ms）
        "min_speech_duration": 250,
        # 最小静默时长（ms）
        "min_silence_duration": 500,
    },
    "xiaozhi": {
        "OTA_URL": "http://127.0.0.1:8003/xiaozhi/ota/",
        "WEBSOCKET_URL": "ws://127.0.0.1:8000/xiaozhi/v1/",
        "WEBSOCKET_ACCESS_TOKEN": "", #（可选）一般用不到这个值
        "DEVICE_ID": "6c:1f:f7:8d:61:b0", #（可选）默认自动生成
        "VERIFICATION_CODE": "", # 首次登陆时，验证码会在这里更新
    },
    "xiaoai": {
        "continuous_conversation_mode": True,
        "exit_command_keywords": ["停止", "退下", "退出", "下去吧"],
        "max_listening_retries": 2,  # 最多连续重新唤醒次数
        "exit_prompt": "再见，主人",
        "continuous_conversation_keywords": ["开启连续对话", "启动连续对话", "我想跟你聊天"]
    },
    # TTS (Text-to-Speech) Configuration
    "tts": {
        "doubao": {
            # 豆包语音合成 API 配置
            # 文档地址: https://www.volcengine.com/docs/6561/1598757?lang=zh
            # 产品地址: https://www.volcengine.com/docs/6561/1871062
            "app_id": "xxxx",         # 你的 App ID
            "access_key": "xxxxxx",       # 你的 Access Key
            "default_speaker": "zh_female_vv_uranus_bigtts",  # 音色 https://www.volcengine.com/docs/6561/1257544?lang=zh
            "audio_format": "pcm",  # 推荐默认值：局域网稳定环境下首音更快、播放更顺
            "stream": True,  # 推荐默认值：边合成边播放，首音延迟更低
        }
    },
    # OpenClaw Configuration
    "openclaw": {
        "url": "ws://127.0.0.1:18789",  # OpenClaw WebSocket 地址
        "token": "your_openclaw_token",  # OpenClaw 认证令牌
        "session_key": "agent:main:open-xiaoai-bridge", # 会话标识
        "identity_path": "/app/openclaw/identity/device.json",  # 设备身份文件路径；容器部署时建议挂载持久化目录
        "tts_speed": 1.0,  # TTS 语速 (0.5-2.0)，仅豆包 TTS 生效，小爱原生 TTS 不支持调速
        "tts_speaker": "xiaoai",  # "xiaoai" = 小爱原生 TTS；填豆包音色 ID 则用豆包 TTS；不设置则使用 tts.doubao.default_speaker
        "response_timeout": 120,  # 等待 OpenClaw agent 响应的超时时间（秒）
        "exit_keywords": ["退出", "停止", "再见"],  # 退出连续对话的关键词
        "rule_prompt": "注意：将结果处理成纯文字版，不要返回任何 markdown 格式，也不要包含任何代码块，并将字数控制在300字以内" # 每次发送消息时自动追加的指令后缀（约束规范）
    },
}
