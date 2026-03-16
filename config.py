import asyncio
import socket
import subprocess
import sys
import time

import requests


async def before_wakeup(speaker, text, source, xiaozhi, xiaoai, app):
    """
    处理收到的用户消息，并决定是否唤醒小智 AI

    - source: 唤醒来源
        - 'kws': 关键字唤醒
        - 'xiaoai': 小爱同学收到用户指令
    """
    if source == "kws":
        # 播放唤醒提示语
        await speaker.play(text="小智来了")
        # 唤醒小智 AI
        return True

    if source == "xiaoai":
        if "小白" in text:
            # 打断原来的小爱同学回复
            await speaker.abort_xiaoai()

            forwarded_text = text + "\n注意：将结果处理成纯文字版， 不要返回任何 markdown 格式，也不要包含任何代码块，并将字数控制在300字以内"
            # 如果想让 Agent 自主调用 TTS 播报结果，可在消息末尾加入提示，引导 Agent 调用 xiaoai-tts skill 播报结果, 可根据自己的使用场景适当调整提示内容
            # forwarded_text = text + "\n\n注意：这条消息是主人通过小爱音箱发送的，他看不到你回复的文字，选一个适合的音色调用 xiaoai-tts skill 将结果播报出来"

            success = await app.send_to_openclaw(forwarded_text)
            if not success:
                import time; time.sleep(2)
                await speaker.play(text="小白未在线")
            return False

        if text == "召唤小智":
            # 打断原来的小爱同学回复
            await speaker.abort_xiaoai()
            # 唤醒小智 AI
            return True


async def after_wakeup(speaker):
    """
    退出唤醒状态
    """
    await speaker.play(url="http://127.0.0.1:8080/bye.wav")

APP_CONFIG = {
    "wakeup": {
        # 自定义唤醒词列表（英文字母要全小写）
        "keywords": [
            "你好小智",
            "小智小智",
            "hi siri"
        ],
        # 静音多久后自动退出唤醒（秒）
        "timeout": 20,
        # 语音识别结果回调
        "before_wakeup": before_wakeup,
        # 退出唤醒时的提示语（设置为空可关闭）
        "after_wakeup": after_wakeup,
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
            "stream": False,  # 流式合成（边合成边播放，首音延迟更低），跑稳定后可设为 True
        }
    },
    # OpenClaw Configuration
    "openclaw": {
        "url": "ws://127.0.0.1:18789",  # OpenClaw WebSocket 地址
        "token": "your_openclaw_token",  # OpenClaw 认证令牌
        "session_key": "main:open-xiaoai-bridge", # 会话标识
        "identity_path": "/app/openclaw/identity/device.json",  # 设备身份文件路径；容器部署时建议挂载持久化目录
        # tts_enabled = True:  服务端自动播放回复，稳定性好、响应快，但只能使用固定音色和语速
        # tts_enabled = False: 由 Agent 主动调用 skills/xiaoai-tts，可灵活控制音色、语速、情感等，但稳定性和响应速度较差
        "tts_enabled": True,  # 是否由服务端自动 TTS 播报 OpenClaw 回复
        "tts_speed": 1.0,  # TTS 语速 (0.5-2.0, 1.0 为正常语速)
        "tts_speaker": "zh_female_vv_uranus_bigtts",  # 可选：自定义音色，不设置则使用 tts.doubao.default_speaker
        "response_timeout": 120,  # 等待 OpenClaw agent 响应的超时时间（秒）
    },
}
