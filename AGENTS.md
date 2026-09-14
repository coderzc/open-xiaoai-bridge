# AGENTS.md

> 小爱音箱与外部 AI 服务（OpenClaw、Home Assistant）的桥接器。
> 接管音箱音频输入输出，实现与第三方 AI / 智能家居的对话。

## 系统架构

详见 [README.md 系统架构](README.md#系统架构)（含 Mermaid 流程图和各模块工作流程）。

## 项目结构

```
open-xiaoai-bridge/
├── main.py                        # 入口：解析环境变量，启动 MainApp
├── config.py                      # 用户配置（唤醒词、路由钩子、TTS、OpenClaw/HomeAssistant 等）
├── core/
│   ├── app.py                     # MainApp 主控制器（单例，管理生命周期）
│   ├── xiaoai.py                  # XiaoAI 设备接入 / 事件桥接
│   ├── xiaoai_conversation.py     # 小爱连续对话策略
│   ├── openclaw.py                # OpenClaw 网关客户端（连接、消息、TTS 播放）
│   ├── openclaw_conversation.py   # OpenClaw 连续对话控制器
│   ├── homeassistant.py           # HomeAssistantManager（Conversation API 客户端）
│   ├── homeassistant_conversation.py  # Home Assistant 连续对话控制器
│   ├── external_conversation.py   # OpenClaw/HomeAssistant 共用的对话循环基类
│   ├── wakeup_session.py          # 唤醒会话路由（KWS/小爱 → OpenClaw/HomeAssistant）
│   ├── ref.py                     # 全局引用注册表（get/set 依赖注入）
│   ├── models/                    # 模型文件（KWS/VAD/ASR，.gitignore 排除）
│   ├── assets/sounds/             # 音效（tts_notify.mp3 等）
│   ├── services/
│   │   ├── speaker.py             # SpeakerManager 音箱硬件控制
│   │   ├── api_server.py          # HTTP REST API（aiohttp）
│   │   ├── audio/
│   │   │   ├── stream.py          # GlobalStream 全局音频流（多路输入广播）
│   │   │   ├── vad/silero.py      # Silero VAD 语音活动检测（ONNX）
│   │   │   ├── kws/sherpa.py      # Sherpa KWS 关键词唤醒
│   │   │   ├── kws/keywords.py    # 唤醒词文件生成（OPENCLAW_ENABLE / homeassistant.enabled 门控）
│   │   │   └── asr/sherpa.py      # Sherpa ASR 离线语音识别（SenseVoice）
│   │   ├── tts/doubao.py          # 豆包 TTS 客户端（火山引擎）
│   │   └── protocols/
│   │       └── typing.py          # 设备状态 / 事件类型定义
│   └── utils/
│       ├── logger.py              # 彩色日志单例（类名 XiaozhiLogger 为历史命名，功能与 XiaoZhi 后端无关）
│       ├── config.py              # ConfigManager（嵌套路径查询、热重载）
│       ├── config_loader.py       # config.py 动态导入
│       ├── base.py                # 基础工具
│       └── file.py                # 文件工具
├── native/                        # Rust PyO3 扩展（maturin 编译）
│   └── src/
│       ├── lib.rs                 # 模块入口：on_output_data, start_server, stop/start_recording, stop/start_playing
│       ├── server.rs              # WebSocket 音频服务器（TCP :4399）
│       ├── python.rs              # Python 回调注册中心（HashMap）
│       ├── macros.rs              # 辅助宏
│       └── tts/                   # TTS 音频处理（流式、PCM 直通、MP3 解码）
├── app/openclaw/                  # OpenClaw 设备身份存储（Ed25519 密钥）
├── skills/xiaoai-tts/             # Agent 工具：通过 HTTP API 控制小爱播放
└── tests/                         # 测试脚本
```

> 本项目当前只保留 **OpenClaw** 与 **Home Assistant** 两条外部对话链路。
> 早期支持过的 XiaoZhi / OpenAI 兼容服务 / QwenPaw 后端已整体移除
> （历史实现可在 git 历史中找到，不再维护）。

## 核心组件

### MainApp (core/app.py)

应用主控制器，单例模式，管理全部服务生命周期。

- `instance(enable_openclaw)` → 单例获取（Home Assistant 无独立开关，见下方说明）
- `run(enable_api_server)` → 启动各服务
- `send_to_openclaw(text, wait_response)` → 发送消息到 OpenClaw（返回 run_id 或回复文本）
- `send_to_openclaw_and_play_reply(text, wait_response)` → 发送并 TTS 播放回复
- `set_openclaw_session_key(session_key)` → 运行时切换 OpenClaw session
- `schedule(callback)` → 主线程任务队列
- `shutdown()` → 优雅关闭

**边界约束**:
- `MainApp` 是业务主循环和设备状态的单一入口
- `device_state` 以 `MainApp` 为准，其他模块通过代理回写，不各自维护平行状态
- `MainApp.loop` 是业务协程的主调度循环
- **Home Assistant 没有 `enable_homeassistant` 参数**：它完全由 `config.py` 的
  `homeassistant.enabled` 驱动。任何"是否启动 VAD/KWS 音频"、"是否预热本地
  ASR"之类的判断，都必须显式读取 `homeassistant.enabled`，不能只看
  `enable_openclaw` ——历史上这里漏过，导致"只开 Home Assistant"时音频服务
  永远不会启动（已在 `core/app.py` / `keywords.py` / `scripts/start.sh` 修复）。

### XiaoAI (core/xiaoai.py)

小爱音箱交互接口，类级变量（classmethod 风格）。

- `init_xiaoai()` → 初始化原生服务，注册事件处理
- `on_event(event)` → 处理小爱事件（RecognizeResult / AudioPlayer）
- `on_input_data(data)` / `on_output_data(data)` → 麦克风 / 扬声器音频回调
- `run_shell(script, timeout)` → 远端 shell 执行
- 内部维护独立 `async_loop`（后台线程），仅用于原生扩展回调和事件桥接

**边界约束**:
- 负责设备接入和事件桥接，不承载连续对话策略
- 连续对话状态放在 `xiaoai_conversation.py`
- `async_loop` 不应承载新的业务状态机

### OpenClawManager (core/openclaw.py)

OpenClaw 网关客户端，管理 WebSocket 连接、消息分发、自动重连、TTS 播放。

- `initialize_from_config()` → 从 config 初始化
- `connect()` → 建立连接（Ed25519 设备身份认证）
- `send(text, wait_response)` → 发送消息，返回 run_id 或回复文本，失败返回 None
- `send_and_play_reply(text, wait_response)` → 发送并 TTS 播放回复
- `is_connected()` / `is_enabled()` → 状态查询

**内部机制**:
- Ed25519 设备身份认证（密钥存储在 `app/openclaw/identity/`）
- WebSocket ping/pong + tick 事件监控连接健康
- 指数退避重连（初始 1s，最大 60s）
- 请求 ID 映射 `_pending: dict[str, asyncio.Future]` 追踪响应
- TTS 播放：`tts_speaker` 为 `"xiaoai"` 时使用小爱原生 TTS，否则使用豆包 TTS（支持流式）
- Rust TTS 播放使用单一活动 `playback_token`：开始新的 Rust TTS 会使旧 token 失效；`stop_tts_playback(token)` 只应由持有该 token 的调用方定向停止自己的播放

**连接参数限制**:
- `client.id`: 必须是 OpenClaw 预定义常量
- `client.mode`: 必须是预定义常量
- `session_key`: 只从 config.py 读取

### HomeAssistantManager (core/homeassistant.py)

Home Assistant Conversation API 客户端，负责与 HA 的
`/api/conversation/process` 交互。

- `send(text, ...)` → 提交一句话给 HA Agent，返回解析后的回复文本
- `reset_conversation()` → 清空本地维护的 `conversation_id`（用于会话重置）
- `start_session()` / `end_session()` → 可选的自定义状态实体维护（`homeassistant.state.enabled`）
- `update_state(...)` → 按配置回写 HA 的状态实体属性（会话轮次、最近一句话等）

**关键坑点**:
- HA 响应中的 `continue_conversation` 字段语义是"HA 自己是否需要追问一句"
  （例如设置计时器缺少时长），**不代表**"是否应保持这次唤醒的连续对话打开"。
  不能直接拿它当作多轮对话循环的退出条件——大多数已执行完的普通指令都会
  返回 `false`。是否连续对话由独立的 `homeassistant.continuous_conversation`
  配置开关控制，见 `homeassistant_conversation.py`。

### HomeAssistantConversationController (core/homeassistant_conversation.py)

Home Assistant 连续对话控制器，继承 `ExternalConversationController`。

- `start()` → 进入对话模式（播放 `intro_prompt`，进入对话循环）
- `stop()` / `is_active()` → 继承自基类
- `_run_one_turn_with_xiaoai_asr()` → 覆盖基类实现，接管小爱原生 ASR 结果，
  一轮对话结束后根据 `continuous_conversation` 开关和 HA 的
  `continue_conversation` 决定是继续监听还是退出
- `_play_tts(text)` → 覆盖基类的豆包/OpenClaw TTS 逻辑，固定使用小爱原生 TTS

**边界约束**:
- 当前只验证过 `input_mode: xiaoai_asr`；`local_asr` 路径依赖
  `HomeAssistantManager` 未实现的 `_send_and_track`/`_wait_response`，走到会报错

### ExternalConversationController (core/external_conversation.py)

OpenClaw 与 Home Assistant 共用的连续对话循环基类。

- `_conversation_loop()` → `while self.active` 主循环，逐轮调用
  `_run_one_turn_with_local_asr()` 或 `_run_one_turn_with_xiaoai_asr()`
  （取决于 `input_mode`），直到某一轮返回 `"exit"`
- `_wait_for_speech(vad)` / `_wait_for_silence(vad)` → 本地 VAD 语音检测
- `consume_xiaoai_recognize_result(...)` → 供 `WakeupSessionManager` 转发小爱原生 ASR 结果
- `_play_tts(text)` → 默认实现（子类可覆盖，如 HomeAssistant 固定用小爱原生 TTS）
- `_stop_recording()` / `_start_recording()` → TTS 播放期间关闭麦克风防回声，播放结束恢复

### WakeupSessionManager (core/wakeup_session.py)

唤醒会话路由，协调 KWS/小爱原生唤醒 → OpenClaw/HomeAssistant 的分发。

- `wakeup(text, source)` → 处理唤醒（调用 `before_wakeup` 钩子，路由到
  OpenClaw 或 Home Assistant）
- `consume_xiaoai_asr_result(...)` → 把小爱原生 ASR 结果转发给当前激活的
  外部对话控制器（Home Assistant 优先，OpenClaw 其次）
- `on_interrupt()` → 小爱唤醒时：cancel OpenClaw/HomeAssistant task、停止设备
  音频播放、恢复录音通道、stop XiaoAI conversation
- `reset_all_sessions()` → 停止所有活跃会话并重置

**路由规则**（`before_wakeup` 返回值）:
- `"openclaw"` → 走 OpenClaw 连续对话
- `"homeassistant"` → 走 Home Assistant 连续对话
- `None` → 不处理（用户自行处理）

**边界约束**:
- 只允许缓存 `on_speech` / `on_silence` 等外部探测信号
- 不要缓存唤醒/中断等控制步骤

### XiaoAIConversationController (core/xiaoai_conversation.py)

小爱自身的连续对话管理。

- `handle_text_command(text, speaker)` → 处理退出 / 连续对话关键词
- `handle_listening_timeout(speaker)` → 超时重试逻辑
- `handle_audio_player_instruction(header_name)` → 检测播放器指令退出
- `handle_playing_status(playing_status, speaker)` → TTS 完成后重新唤醒

**边界约束**:
- 小爱自身的连续对话和外部唤醒 / 会话超时是两套独立机制
- 只有在"小爱连续对话确实激活"时才允许停止

### SpeakerManager (core/services/speaker.py)

音箱硬件控制。

- `play(text, url, buffer, blocking, timeout)` → 播放文字 / URL / PCM 缓冲
- `stop_device_audio()` → 停止设备上的播放链路（阻塞 TTS / 非阻塞 TTS / PCM），并重启 PCM 播放通道
- `wake_up(awake, silent)` → 唤醒 / 休眠小爱
- `abort_xiaoai()` → 中断小爱当前操作
- `ask_xiaoai(text, silent)` → 让小爱执行指令
- `run_shell(command, timeout)` → RPC shell

**边界约束**:
- `stop_device_audio()` 只负责"停播放"，不负责恢复录音；`start_recording()` 属于会话层恢复逻辑，应由 `WakeupSessionManager` / `ExternalConversationController` 等上层按场景决定

### APIServer (core/services/api_server.py)

HTTP REST API 服务器（aiohttp），端口可配（默认 9092）。

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/play/text` | POST | 播放文本 |
| `/api/play/url` | POST | 播放 URL |
| `/api/play/file` | POST | 播放本地文件 |
| `/api/status` | GET | 获取设备状态 |
| `/api/wakeup` | POST | 唤醒设备 |
| `/api/interrupt` | POST | 中断播放 |
| `/api/health` | GET | 健康检查 |
| `/api/tts/doubao` | POST | Doubao TTS 合成 |
| `/api/tts/doubao_voices` | GET | 获取音色列表 |

### 音频处理链

| 模块 | 文件 | 职责 |
|------|------|------|
| GlobalStream | `audio/stream.py` | 多路输入广播（模拟 PyAudio API） |
| VAD | `audio/vad/silero.py` | Silero ONNX 语音活动检测 |
| KWS | `audio/kws/sherpa.py` | Sherpa ONNX 关键词唤醒（信心度 2.0，阈值 0.2） |
| ASR | `audio/asr/sherpa.py` | Sherpa SenseVoice 离线语音识别（懒加载，INT8 量化） |
| TTS | `tts/doubao.py` | 豆包 TTS（流式/一次性，PCM/MP3 自适应） |

### Rust 原生扩展 (native/)

通过 maturin + PyO3 编译的 `open_xiaoai_server` Python 模块。

| 文件 | 职责 |
|------|------|
| `lib.rs` | 模块入口：`on_output_data`, `start_server`, `stop/start_recording`, `stop/start_playing`, `run_shell` |
| `server.rs` | TCP :4399 WebSocket 服务器，处理音频流和事件路由 |
| `python.rs` | Python 回调注册中心（`register_fn` / `call_fn`），跨语言调用 |
| `tts/` | TTS 音频处理：HTTP 流式请求、MP3 解码、PCM 直通 |

## 运行模式

### 模式 1: 仅小爱（默认）
```bash
uv run main.py
```
- 不启动 KWS/VAD 初始化
- `core/services/audio/kws/keywords.py` 在此模式下应直接退出成功

### 模式 2: OpenClaw
```bash
OPENCLAW_ENABLE=1 uv run main.py
```
- 小爱指令拦截 → 转发到 OpenClaw → TTS 播放结果
- OpenClaw 连续对话：VAD/小爱 ASR → OpenClaw → TTS 循环
- 退出关键词：config `openclaw.exit_keywords`

### 模式 3: Home Assistant
```bash
# 没有独立环境变量，改在 config.py 中设置：
# APP_CONFIG["homeassistant"]["enabled"] = True
uv run main.py
```
- 小爱/KWS 唤醒 → 转发到 HA Conversation API → 小爱原生 TTS 播放结果
- 是否连续对话：config `homeassistant.continuous_conversation`
- 退出关键词：config `homeassistant.exit_keywords`

### 模式 4: OpenClaw + Home Assistant（混合）
```bash
OPENCLAW_ENABLE=1 uv run main.py
# 同时在 config.py 中设置 homeassistant.enabled = True
```
- config.py `before_wakeup` 按唤醒词路由到 OpenClaw 或 Home Assistant 连续对话

### 启用 API Server
```bash
API_SERVER_ENABLE=1 uv run main.py
```

## 开发规范

### 代码风格
- 中文注释和文档字符串
- 英文 commit message
- 类型提示: `dict[str, asyncio.Future]`

### 异步编程
- 所有 I/O 使用 `async/await`
- 线程安全使用 `asyncio.run_coroutine_threadsafe()`
- `MainApp.loop` 是业务协程主循环
- `XiaoAI.async_loop` 仅用于原生扩展回调桥接，不挂新业务状态机

### 日志规范
- 所有日志必须带模块标识：通过 `module=` 参数或 `[Module]` 前缀
- 使用 `core.utils.logger.logger`，禁止裸 `print`
- 调试输出用 `DEBUG` 级别，不污染 `INFO`
- 消息体不要重复模块名（模块名已在日志前缀中）
- 唯一允许的裸输出：启动 ASCII banner

### 全局引用 (ref.py)
- `set_app/get_app`, `set_xiaoai/get_xiaoai`
- `set_vad/get_vad`, `set_kws/get_kws`, `set_speaker/get_speaker`

### 兼容约束
- `CLI` 环境变量不再作为功能开关，不要引入依赖 `CLI` 的运行时分支
- `OPENCLAW_ENABLE=0` 且 `homeassistant.enabled=False` 时必须允许跳过 KWS 初始化
- `scripts/start.sh` 在仅小爱模式下不应检查 `core/models/` 下的模型文件
- Home Assistant 相关的功能开关一律读 `config.py` 的
  `homeassistant.*`，不要为它引入新的环境变量（保持与 OpenClaw 不对称是
  故意的设计，不是遗漏）

## 测试

```bash
# 无音箱流式冒烟测试
python3 tests/test_tts_stream.py

# 比较长文本 mp3/pcm 流式时延
python3 tests/test_tts_latency.py --formats mp3,pcm --rounds 3 --repeat 8

# OpenClaw 连通性测试
python3 tests/test_openclaw_live_connectivity.py

# 唤醒词生成门禁 + KWS 路由单测（无需硬件/网络）
python3 -m pytest tests/test_wakeup_keywords.py -v
```

## 音箱设备控制命令

小爱音箱（LX06 等）基于 OpenWrt + busybox，设备端命令和行为如下：

### 音频播放通道

音箱上有多条独立的音频播放通道，中断时需要分别处理：

| 通道 | 进程/服务 | 触发方式 | 中断方式 |
|------|-----------|---------|---------|
| PCM 直通 | `aplay` | `open_xiaoai_server.start_playing()` → WebSocket stream | `open_xiaoai_server.stop_playing()` |
| 阻塞 TTS | `tts_play.sh` → `miplayer -f <file>` | `speaker.play(blocking=True)` | `killall tts_play.sh miplayer` |
| 非阻塞 TTS | `mibrain_service` (内部播放) | `speaker.play(blocking=False)` → `ubus call mibrain text_to_speech` | `mphelper pause`（不一定可靠） |
| 媒体播放器 | `mediaplayer` (系统守护进程) | `ubus call mediaplayer player_play_url` | `mphelper pause` / `ubus call mediaplayer player_play_operation '{"action":"pause"}'` |

### tts_play.sh 工作流程

`/usr/sbin/tts_play.sh` 是设备上的阻塞 TTS 脚本，内部流程：
1. `mphelper pause` — 暂停当前播放
2. `ubus call mibrain text_to_speech '{"text":"...","save":1}'` — 生成音频文件到 `/tmp/tts/`
3. `miplayer -f <path>` — 播放音频文件（子进程）
4. `rm <path>` — 清理临时文件

**关键注意事项**：
- 杀掉 `tts_play.sh` **不会**自动杀掉子进程 `miplayer`，必须同时 `killall miplayer`
- `miplayer` 是一次性播放器（非守护进程），杀掉后不影响后续 TTS 调用
- busybox 的 `pkill` 无法匹配到 `miplayer`，必须用 `killall`

### 录音通道

| 操作 | 命令 | 说明 |
|------|------|------|
| 停止录音 | `open_xiaoai_server.stop_recording()` | 杀掉设备端 `arecord` 进程，麦克风静音 |
| 恢复录音 | `open_xiaoai_server.start_recording()` | 重启 `arecord`，音频数据恢复流入 `GlobalStream` |

**注意**：OpenClaw / Home Assistant 对话中 TTS 播放时会 `stop_recording` 防止回声。如果在此期间触发中断（"小爱同学"），必须在中断处理中调用 `start_recording` 恢复录音，否则 KWS 将因无音频数据而永久失效。

### on_interrupt 中断处理要点

`on_interrupt()` 触发时（用户喊"小爱同学"），需要完成以下步骤：
1. Cancel OpenClaw / Home Assistant asyncio task
2. 让对应的 conversation controller 自己停止当前 TTS（使用自己持有的 `playback_token`）
3. `SpeakerManager.stop_device_audio()` — 停止阻塞 TTS / 非阻塞 TTS / PCM，并重置 PCM 通道
4. `start_recording` — 恢复录音（KWS 依赖此通道）
5. `XiaoAI.stop_conversation()` — 停止连续对话

### 不可用的中断方式

以下方式在实践中验证**不可靠或有副作用**：
- `abort_xiaoai()`（重启 `mico_aivs_lab`）— 会导致小爱整体不可用，恢复需 1-2 秒
- `pkill miplayer` — busybox `pkill` 无法匹配 `miplayer` 进程名
- `ubus call mediaplayer player_play_operation '{"action":"pause"}'` — 对 `mibrain text_to_speech` 触发的播放无效

### 相关讨论

- [open-xiaoai#36](https://github.com/idootop/open-xiaoai/issues/36) — 小爱 TTS 打断方案讨论

## 参考资源

- 项目主页: https://github.com/coderzc/open-xiaoai-bridge
- 刷机教程: https://github.com/idootop/open-xiaoai/blob/main/docs/flash.md
- Client 端补丁: https://github.com/idootop/open-xiaoai/blob/main/packages/client-rust/README.md
