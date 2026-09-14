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
        "openclaw"      — 进入 OpenClaw 连续对话流程
        "homeassistant" — 进入 Home Assistant 连续对话流程
        None            — 不做额外处理（可在此自行调用 app.send_to_openclaw 等）

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
        if "龙虾" in text:
            await speaker.play(text="龙虾来了")
            return "openclaw"

        # Route to Home Assistant by wake word
        if "小▁p" in text:
            await speaker.play(text="我来了")
            return "homeassistant"

        return None

    if source == "xiaoai":
        # --- 示例四：小爱指令按用户名路由到不同 Session ---
        # if text == "召唤小美":
        #     app.set_openclaw_session_key("agent:xiaomei:open-xiaoai-bridge")
        #     await speaker.abort_xiaoai()
        #     return "openclaw"
      #  await speaker.play(text="我来了，请问有什么需要帮忙的吗？")
        await speaker.abort_xiaoai()
        return None
        
async def after_wakeup(speaker, source=None, session_key=None):
    """
    退出唤醒状态

    - source: 退出来源
        - 'openclaw': OpenClaw 连续对话退出
        - 'homeassistant': Home Assistant 连续对话退出
    - session_key: 当前 OpenClaw 后端 session_key
        可据此区分是哪个 Agent 退出，例如播放不同的退出提示语
    """
    if source == "openclaw":
        # 示例：退出 OpenClaw 时播放服务端本地结束语
        # await speaker.play(server_file="/path/to/openclaw_bye.wav")

        # 示例：按 agentId 区分退出提示语
        # 所有后端的 session_key 已统一为 agent:<agentId>:<rest> 格式，第二段即 agentId。
        # 仍建议做越界防护，以兼容自定义的非标准 session_key。
        # parts = session_key.split(":") if session_key else []
        # agent_id = parts[1] if len(parts) > 1 else None
        # if agent_id == "assistant":
        #     await speaker.play(text="助手，再见")
        # elif agent_id == "xiaomei":
        #     await speaker.play(text="小美，再见")
        # else:
        #     await speaker.play(text="再见")
        await speaker.play(text="龙虾，再见")
    if source == "homeassistant":
        await speaker.play(text="我先退下了")
           

APP_CONFIG = {
    "wakeup": {
        # 自定义唤醒词列表（英文字母要全小写）
        "keywords": [
            "hi open claw",
            "你好龙虾",
            "龙虾你好",
            "你好小p",
            "小p你好",
            "小p",
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
    "audio_input": {
        # Input gain multiplier before VAD/KWS/ASR. Use 1.0 to disable.
        "gain": 1.0,
    },
    "asr": {
        # 支持 "sense_voice"（默认）、"paraformer"、"fire_red_asr" 或 "doubao"
        "model": "sense_voice",
        # 是否优先使用 INT8 量化模型（仅本地模型生效）
        "int8": True,
        # 可选：显式指定 core/models/ 下的模型目录名（仅本地模型生效）
        # "model_dir": "",
        "doubao": {
            # "standard": 录音文件识别标准版，调用 /submit + /query
            # "flash": 录音文件极速版，调用 /recognize/flash
            "mode": "standard",
            "app_key": "你的 App Key",
            "access_key": "你的 Access Key",
            # 火山 X-Api-Resource-Id：
            # standard 可选：
            #   "volc.bigasr.auc"  - 豆包录音文件识别模型 1.0
            #   "volc.seedasr.auc" - 豆包录音文件识别模型 2.0
            # flash 可选：
            #   "volc.bigasr.auc_turbo" - 录音文件极速版
            "resource_id": "volc.seedasr.auc",
            "language": "",
            "submit_timeout": 10,
            "query_timeout": 10,
            "poll_interval": 0.5,
            "max_wait_seconds": 20,
        },
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
        # 输入模式：
        #   - "local_asr": 现有链路，使用本地 VAD + SherpaASR
        #   - "xiaoai_asr": 实验链路，唤醒小爱后接管原生 ASR 结果给 OpenClaw
        "input_mode": "local_asr",
        # session_key 格式：agent:<agentId>:<rest>
        #   agentId: OpenClaw 中配置的 Agent ID（默认为 main）
        #   rest:    会话标识，可自由命名，用于区分不同来源/场景 （默认为 open-xiaoai-bridge)
        # 也可在运行时动态切换（下一条消息即刻生效，无需重连）：
        #   app.set_openclaw_session_key("agent:assistant:open-xiaoai-bridge")
        "session_key": "agent:main:open-xiaoai-bridge",
        "identity_path": "/app/openclaw/identity/device.json",  # 设备身份文件路径；容器部署时建议挂载持久化目录
        "tts_speed": 1.0,  # TTS 语速 (0.5-2.0)，仅豆包 TTS 生效，小爱原生 TTS 不支持调速
        "tts_speaker": "xiaoai",  # "xiaoai" = 小爱原生 TTS；填豆包音色 ID 则用豆包 TTS；不设置则使用 tts.doubao.default_speaker
        # 可按 agentId 单独覆盖音色，优先级高于 tts_speaker
        # agentId 来自 session_key，格式为：agent:<agentId>:<rest>
        # 示例：
        # "agent_tts_speakers": {
        #     "assistant": "zh_female_vv_uranus_bigtts",
        #     "xiaomei": "zh_female_shuangkuaisisi_moon_bigtts",
        #     "butler": "xiaoai",
        # },
        "agent_tts_speakers": {},
        "response_timeout": 120,  # 等待 OpenClaw agent 响应的超时时间（秒）
        "exit_keywords": ["退出", "停止", "再见"],  # 退出连续对话的关键词
        # rule_prompt: 用于「自动播放」和「连续对话」场景
        #   - send_to_openclaw_and_play_reply() 会自动追加
        #   - OpenClawConversationController 会自动追加
        "rule_prompt": "注意：将结果处理成纯文字版，不要返回任何 markdown 格式，也不要包含任何代码块，并将字数控制在300字以内",
        # rule_prompt_for_skill: 用于「Agent 自主播报」场景（方式三）
        #   - send_to_openclaw() 会自动追加
        #   - 告诉 Agent 需要调用 xiaoai-tts skill 来播报，因为服务端不会自动播放
        "rule_prompt_for_skill": "注意：这条消息是主人通过小爱音箱发送的，他看不到你回复的文字，调用 `xiaoai-tts` skill 播报出来。字数控制在300字以内"
    },
    # Home Assistant Configuration
    "homeassistant": {
        # 是否启用
        #
        # 注意：Home Assistant 没有独立的环境变量开关（不同于
        # OPENCLAW_ENABLE），完全由这里的 enabled 值控制，
        # 包括是否启动本地 VAD/KWS 音频服务。
        "enabled": False,

        # Home Assistant 地址
        "base_url": "http://192.168.100.80:8123",

        # Home Assistant Long-Lived Access Token
        "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI0NGM5MTU2NjNlOWQ0YThiOTdjMWM2ZWZiZTJiMzY3OCIsImlhdCI6MTc4ODY2NTUzNCwiZXhwIjoyMTA0MDI1NTM0fQ.mMEI6pXEV-oVo78BdHHrGnGjvJHly5k6CnMLj8JjAq4",

        # Conversation Agent
        #
        # 例如：
        # conversation.home_assistant
        # conversation.qwen
        #
        # 留空则使用 HA 默认 Agent
        "agent_id": "conversation.ollama_conversation",

        # 输入模式
        #
        # local_asr:
        #     本地 VAD + ASR
        #
        # xiaoai_asr:
        #     使用小爱原生 ASR
        "input_mode": "xiaoai_asr",

        # HTTP 请求超时时间
        "response_timeout": 120,

        # 等待小爱原生 ASR 的超时时间
        "timeout": 30,

        # 是否保持 Home Assistant conversation_id
        "keep_conversation": True,

        # 是否开启连续对话
        #
        # True（默认）：
        #     唤醒一次后可连续下达多条指令
        #     （例如"打开客厅灯" -> "再打开卧室灯" -> "调到50%亮度"），
        #     直到静音超过 timeout 秒，或说出 exit_keywords
        #     中的退出词才结束对话。
        #
        # False：
        #     每次唤醒只执行一条指令就结束对话
        #     （更接近"一次唤醒一次命令"的传统语音助手体验）。
        #     注意：即使关闭，如果 HA 的 Agent 自身需要追问
        #     （例如设置计时器时缺少时长，返回了
        #     continue_conversation=True），仍会等待一轮
        #     用户回答，不会打断 HA 自己的澄清流程。
        "continuous_conversation": True,

        # 本地 session 标识
        "session_key": "homeassistant:open-xiaoai-bridge",

        # Home Assistant Conversation 参数
        "language": "zh-CN",

        # 连续对话退出关键词
        "exit_keywords": [
            "退出",
            "停止",
            "再见",
            "滚",
            "退下",
            "结束对话",
        ],

        # TTS
        #
        # 当前实现：
        # xiaoai = 使用小爱原生 TTS
        "tts_speed": 1.0,
        "tts_speaker": "xiaoai",

        # 进入 Home Assistant 模式时的提示语
        #
        # 留空 = 不播放
        "intro_prompt": "",

        # 退出 Home Assistant 模式时的提示语
        "exit_prompt": "好的，再见",

        # 发给 Home Assistant Agent 前附加的规则
        "rule_prompt": (
            "注意：这是通过小爱音箱进行的语音对话。"
            "请直接返回适合语音播报的纯文字内容，"
            "不要使用 Markdown、代码块或特殊格式，"
            "回答尽量简洁。"
        ),

        # 自定义状态
        "state": {
            # 是否维护 HA 状态实体
            "enabled": False,

            # 你的现有实体
            "entity_id": "sensor.xiaomi_lx06_7e15_conversation",

            # 是否更新用户最后一句话
            "update_user_text": True,

            # 是否更新 AI 最后回复
            "update_response": True,

            # 是否写入这些 attributes
            "attributes": {
                "mode": True,
                "conversation_id": True,
                "turn_count": True,
                "last_user_text": True,
                "last_response": True,
                "source": True,
                "wake_word": True,
                "updated_at": True,
            },
        },
    },
}
