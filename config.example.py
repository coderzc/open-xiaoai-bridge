import asyncio
import socket
import subprocess
import sys
import time
import uuid

import requests


# 每次唤醒生成独立 Session，对话互不干扰
# 适合以下场景：
#   - "提问 → 回答"式交互，不需要 Agent 记住上下文
#   - 长期使用同一 Session 导致 Agent 上下文窗口堆积过长，影响响应质量和速度
# def new_session_key():
#     return f"agent:main:session-{uuid.uuid4().hex[:8]}"


async def before_wakeup(speaker, text, source, app):
    """
    处理收到的用户消息，并决定是否唤醒 AI。

    参数：
        speaker : SpeakerManager，可调用 play/abort_xiaoai/wake_up 等方法
        text    : 识别到的文字内容
        source  : 唤醒来源
                    'kws'    — 本地关键词唤醒（用户说了唤醒词）
                    'xiaoai' — 小爱同学收到用户语音指令
        app     : MainApp 实例，可调用 send_to_openclaw 等方法

    返回值：
        "openclaw" — 进入 OpenClaw 连续对话流程
        "xiaozhi"  — 进入小智 AI 流程
        None       — 不做额外处理（可在此自行调用 app.send_to_openclaw 等）

    ---
    动态切换 session_key：
        每次进入此函数前，框架会自动将 session_key 重置为配置文件中的默认值。
        如需路由到其他 Agent，在 return "openclaw" 之前调用：
            app.set_openclaw_session_key("agent:main:xxx")
        不调用则自动使用 openclaw.session_key 的默认值，无需手动重置。
    """
    if source == "kws":
        # --- 示例一：按唤醒词路由到不同 Agent ---
        # AGENT_SESSIONS = {
        #     "龙虾": "agent:assistant:open-xiaoai-bridge",  # 说"你好龙虾" → 路由到 assistant Agent
        #     "小美": "agent:xiaomei:open-xiaoai-bridge",    # 说"你好小美" → 路由到 xiaomei Agent
        #     "管家": "agent:butler:open-xiaoai-bridge",     # 说"你好管家" → 路由到 butler Agent
        # }
        # for keyword, session_key in AGENT_SESSIONS.items():
        #     if keyword in text:
        #         app.set_openclaw_session_key(session_key)
        #         await speaker.play(text=f"{keyword}来了")
        #         return "openclaw"

        # --- 示例二：每次唤醒生成独立 Session ---
        # if "龙虾" in text:
        #     app.set_openclaw_session_key(new_session_key())
        #     await speaker.play(text="龙虾来了")
        #     return "openclaw"

        # --- 示例三：进入 OpenClaw 前播放服务端本地开场白 ---
        # if "龙虾" in text:
        #     await speaker.play(server_file="/path/to/openclaw_intro.wav")
        #     return "openclaw"

        # Route to OpenClaw Agent by wake word
        if "螃蟹" in text:
            await speaker.play(text="螃蟹来了")
            return "openclaw"

        if "小智" in text:
            await speaker.play(text="小智来了")
            return "xiaozhi"

        return None

    if source == "xiaoai":
        # --- 示例四：小爱指令按用户名路由到不同 Session ---
        # if text == "召唤小美":
        #     app.set_openclaw_session_key("agent:xiaomei:open-xiaoai-bridge")
        #     await speaker.abort_xiaoai()
        #     return "openclaw"

        if text == "召唤螃蟹":
            await speaker.abort_xiaoai()
            return "openclaw"  # OpenClaw continuous conversation

        if text == "召唤小智":
            await speaker.abort_xiaoai()
            return "xiaozhi"  # XiaoZhi AI

        if "请" in text:
            await speaker.abort_xiaoai()
            # One-shot: send to OpenClaw and play the reply via TTS
            await app.send_to_openclaw_and_play_reply(text.replace("请", ""))
            return None  # No further handling by the framework

        if "告诉螃蟹" in text:
            await speaker.abort_xiaoai()
            # Fire-and-forget: let the Agent decide when/how to reply
            await app.send_to_openclaw_and_play_reply(text.replace("告诉螃蟹", ""))
            return None


async def after_wakeup(speaker, source=None, session_key=None):
    """
    退出唤醒状态

    - source: 退出来源
        - 'xiaozhi': 小智对话超时退出
        - 'openclaw': OpenClaw 连续对话退出
    - session_key: 当前 OpenClaw session_key（仅 source='openclaw' 时传入）
        可据此区分是哪个 Agent 退出，例如播放不同的退出提示语
    """
    if source == "openclaw":
        # 示例：退出 OpenClaw 时播放服务端本地结束语
        # await speaker.play(server_file="/path/to/openclaw_bye.wav")

        # 示例：按 agentId 区分退出提示语
        # session_key 格式：agent:<agentId>:<rest>，第二段即 agentId
        # agent_id = session_key.split(":")[1] if session_key else None
        # if agent_id == "assistant":
        #     await speaker.play(text="助手，再见")
        # elif agent_id == "xiaomei":
        #     await speaker.play(text="小美，再见")
        # else:
        #     await speaker.play(text="再见")
        await speaker.play(text="再见")
    if source == "xiaozhi":
        await speaker.play(text="再见")

APP_CONFIG = {
    "wakeup": {
        # 唤醒检测方式：
        #   - "local": 使用本地 Sherpa KWS 模型直接做关键词唤醒
        #   - "doubao": 使用基于 ASR 的唤醒检测（可选本地 Sherpa / OpenAI 兼容 STT）
        "provider": "doubao",
        # 自定义唤醒词列表（英文字母要全小写）
        "keywords": [
            "你好小智",
            "小智小智",
            "hi open claw",
            "你好螃蟹",
            "螃蟹你好",
            "螃蟹",
            "螃蟹螃蟹",
        ],
        # 基于 ASR 的唤醒检测参数（仅 wakeup.provider="doubao" 时生效）
        "doubao": {
            # 唤醒识别模式：
            #   - "rolling_window": 现有方式，滚动窗口周期性送检
            #   - "utterance_end": 等一整段语音结束后再送检一次
            "mode": "utterance_end",
            # 唤醒词识别阶段使用的 ASR 引擎：
            #   - "sherpa": 本地离线识别
            #   - "openai_compatible": 兼容 OpenAI /audio/transcriptions 的 STT（当前可走代理到豆包）
            "asr_provider": "openai_compatible",
            # rolling_window 模式：每次送去识别的音频窗口大小（毫秒）
            "window_ms": 3500,
            # rolling_window 模式：两次检测之间的最小间隔（毫秒）
            "check_interval_ms": 1800,
            # 一次命中后，下次允许再次触发的最小间隔（毫秒）
            "min_gap_ms": 1500,
            # 小于该 PCM 字节数的音频直接跳过，不送识别
            "min_audio_bytes": 20000,
            # 单次唤醒送检允许的最大音频时长（毫秒）；超过后直接丢弃，不送识别。默认 2 秒。
            "max_audio_ms": 2000,
            # 单次唤醒送检允许的最大 PCM 字节数；超过后直接丢弃，不送识别。
            "max_audio_bytes": 64000,
            # utterance_end 模式：连续静音达到多少帧后视为这一整段语音已经结束
            "utterance_end_silence_frames": 12,
            # utterance_end 模式：单段语音最多累计多长时间（毫秒）
            "utterance_max_ms": 5000,
        },
        # 静音多久后自动退出唤醒（秒）
        "timeout": 20,
        # 语音识别结果回调
        "before_wakeup": before_wakeup,
        # 退出唤醒时的提示语（设置为空可关闭）
        "after_wakeup": after_wakeup,
    },
    "kws": {
        # 唤醒词置信度加成（越高越难误触发，越低越灵敏）
        "keywords_score": 0.1,
        # 唤醒词检测阈值（越低越灵敏，越高越难触发）
        "keywords_threshold": 0.1,
        # 唤醒词检测时的最小静默时长（ms），静默超过该时长则判定为说完
        "min_silence_duration": 480,
    },
    "vad": {
        # 语音检测阈值（0-1，越小越灵敏）
        "threshold": 0.10,
        # 最小语音时长（ms）
        "min_speech_duration": 250,
        # 最小静默时长（ms）
        "min_silence_duration": 1000,
    },
    "asr": {
        # 默认 ASR provider：
        #   - "sherpa": 本地离线识别
        #   - "openai_compatible": 调用兼容 OpenAI /audio/transcriptions 的在线识别接口
        # 这是全局默认值；若某个场景在 contexts 中单独配置，则以场景配置为准
        "provider": "openai_compatible",
        # 主 provider 失败后的回退 provider（可选）
        "fallback_provider": None,
        # 按场景分别指定 ASR provider
        "contexts": {
            # 唤醒后的内容识别（即进入连续对话后说的话）
            "conversation": {
                # 内容识别引擎：
                #   - "sherpa": 本地离线识别
                #   - "openai_compatible": 兼容 OpenAI STT 的在线识别（当前可走代理到豆包）
                "provider": "openai_compatible",
                # 该场景识别失败后的回退引擎（可选）
                "fallback_provider": None,
                # 唤醒后内容识别阶段的输入增益（同时适用于本地 Sherpa 和 OpenAI 兼容 STT）
                "input_gain": 8.0,
                # 唤醒后单轮内容识别允许录制的最大时长（毫秒）；超过后直接丢弃，不送识别。默认 30 秒。
                "max_speech_ms": 30000,
                # 唤醒后单轮内容识别允许缓存的最大 PCM 字节数；超过后直接丢弃，不送识别。
                "max_audio_bytes": 960000,
            },
            # 唤醒词识别（仅 wakeup.provider="doubao" 时生效；若同时配置 wakeup.doubao.asr_provider，则优先使用后者）
            "wakeup": {
                # 唤醒词识别引擎：
                #   - "sherpa": 本地离线识别
                #   - "openai_compatible": 兼容 OpenAI STT 的在线识别（当前可走代理到豆包）
                "provider": "openai_compatible",
                # 该场景识别失败后的回退引擎（可选）
                "fallback_provider": None,
                # 唤醒阶段送检前的输入增益（仅影响基于 ASR 的唤醒，不影响唤醒后内容识别）
                "input_gain": 8.0,
            },
        },
        # ASR 模型名：
        #   - provider=sherpa 时：本地模型名（如 "sense_voice" / "paraformer"）
        #   - provider=openai_compatible 时：上游兼容接口接收的 model 字段
        "model": "bigmodel",
        # 在线识别超时（秒）
        "timeout": 20,
        # 文本替换规则（本地/在线都会生效）
        "replacements": {},
        # OpenAI 兼容 STT 配置
        "openai_compatible": {
            # 兼容 OpenAI 语音识别接口的基础地址
            "base_url": "http://127.0.0.1:8787",
            # 兼容接口所需的 Bearer Token；本地代理场景可填占位值
            "api_key": "YOUR_OPENAI_COMPATIBLE_API_KEY",
        },
    },
    "audio_debug": {
        # 是否开启音频调试模式：
        #   - False：不保存调试 wav
        #   - True：按下面各场景设置保存调试 wav
        "enabled": True,
        "contexts": {
            "wakeup": {
                # 是否保存唤醒词识别阶段的 wav
                "save_wav": True,
                # 唤醒词阶段调试 wav 保存目录
                "dir": "/app/core/debug_wakeup_audio",
                # 最多保留多少条唤醒词阶段 wav，超过后删除更旧文件
                "max_files": 10,
            },
            "conversation": {
                # 是否保存唤醒后内容识别阶段的 wav
                "save_wav": True,
                # 唤醒后内容识别阶段调试 wav 保存目录
                "dir": "/app/core/debug_conversation_audio",
                # 最多保留多少条内容识别阶段 wav，超过后删除更旧文件
                "max_files": 10,
            },
        },
    },
    "xiaozhi": {
        "OTA_URL": "http://127.0.0.1:8003/xiaozhi/ota/",
        "WEBSOCKET_URL": "ws://127.0.0.1:8000/xiaozhi/v1/",
        "WEBSOCKET_ACCESS_TOKEN": "",
        "DEVICE_ID": "YOUR_XIAOZHI_DEVICE_ID",
        "VERIFICATION_CODE": "",
    },
    "xiaoai": {
        "continuous_conversation_mode": True,
        "exit_command_keywords": ["停止", "退下", "退出", "下去吧"],
        "max_listening_retries": 2,
        "exit_prompt": "再见，主人",
        "continuous_conversation_keywords": ["开启连续对话", "启动连续对话", "我想跟你聊天"]
    },
    # TTS (Text-to-Speech) Configuration
    "tts": {
        "doubao": {
            # 豆包语音合成 API 配置
            # 文档地址: https://www.volcengine.com/docs/6561/1598757?lang=zh
            # 产品地址: https://www.volcengine.com/docs/6561/1871062
            "app_id": "YOUR_DOUBAO_APP_ID",
            "access_key": "YOUR_DOUBAO_ACCESS_KEY",
            "default_speaker": "saturn_zh_female_keainvsheng_tob",
            "audio_format": "pcm",
            "stream": True,
        }
    },
    # OpenClaw Configuration
    "openclaw": {
        "url": "ws://127.0.0.1:18789",
        "token": "YOUR_OPENCLAW_TOKEN",
        # 输入模式：
        #   - "local_asr": 现有链路，使用本地 VAD + SherpaASR
        #   - "xiaoai_asr": 实验链路，唤醒小爱后接管原生 ASR 结果给 OpenClaw
        "input_mode": "local_asr",
        # session_key 格式：agent:<agentId>:<rest>
        #   agentId: OpenClaw 中配置的 Agent ID（默认为 main）
        #   rest:    会话标识，可自由命名，用于区分不同来源/场景 （默认为 open-xiaoai-bridge)
        # 也可在运行时动态切换（下一条消息即刻生效，无需重连）：
        #   app.set_openclaw_session_key("agent:assistant:open-xiaoai-bridge")
        "session_key": "agent:speaker:open-xiaoai-bridge",
        "identity_path": "/app/openclaw/identity/device.json",
        "tts_speed": 1.0,
        "tts_speaker": "YOUR_TTS_SPEAKER_OR_XIAOAI",
        # 可按 agentId 单独覆盖音色，优先级高于 tts_speaker
        # agentId 来自 session_key，格式为：agent:<agentId>:<rest>
        # 示例：
        # "agent_tts_speakers": {
        #     "assistant": "zh_female_vv_uranus_bigtts",
        #     "xiaomei": "zh_female_shuangkuaisisi_moon_bigtts",
        #     "butler": "xiaoai",
        # },
        "agent_tts_speakers": {},
        "response_timeout": 120,
        "exit_keywords": ["退出", "停止", "再见"],
        # rule_prompt: 用于「自动播放」和「连续对话」场景
        #   - send_to_openclaw_and_play_reply() 会自动追加
        #   - OpenClawConversationController 会自动追加
        "rule_prompt": "在客厅，请返回300字以内的纯文字信息",
        # rule_prompt_for_skill: 用于「Agent 自主播报」场景（方式三）
        #   - send_to_openclaw() 会自动追加
        #   - 告诉 Agent 需要调用 xiaoai-tts skill 来播报，因为服务端不会自动播放
        "rule_prompt_for_skill": "注意：这条消息是主人通过小爱音箱发送的，他看不到你回复的文字，调用 `xiaoai-tts` skill 播报出来。字数控制在300字以内"
    },
}
