# AGENTS.md

> 小爱音箱与外部 AI 服务（小智 AI、OpenClaw）的桥接器。
> 接管音箱音频输入输出，实现与第三方 AI 的对话。

## 系统架构

详见 [README.md 系统架构](README.md#系统架构)（含 Mermaid 流程图和各模块工作流程）。

## 项目结构

```
open-xiaoai-bridge/
├── main.py                        # 入口：解析环境变量，启动 MainApp
├── config.py                      # 用户配置（唤醒词、路由钩子、TTS、OpenClaw 等）
├── core/
│   ├── app.py                     # MainApp 主控制器（单例，管理生命周期）
│   ├── xiaoai.py                  # XiaoAI 设备接入 / 事件桥接
│   ├── xiaoai_conversation.py     # 小爱连续对话策略
│   ├── xiaozhi.py                 # 小智 AI WebSocket 协议客户端
│   ├── openclaw.py                # OpenClaw 网关客户端（连接、消息、TTS 播放）
│   ├── openclaw_conversation.py   # OpenClaw 连续对话循环（VAD → ASR → Agent → TTS）
│   ├── wakeup_session.py          # 小智唤醒会话状态机
│   ├── ref.py                     # 全局引用注册表（get/set 依赖注入）
│   ├── models/                    # 模型文件（KWS/VAD/ASR，.gitignore 排除）
│   ├── assets/sounds/             # 音效（tts_notify.mp3 等）
│   ├── services/
│   │   ├── speaker.py             # SpeakerManager 音箱硬件控制
│   │   ├── api_server.py          # HTTP REST API（aiohttp）
│   │   ├── audio/
│   │   │   ├── stream.py          # GlobalStream 全局音频流（多路输入广播）
│   │   │   ├── codec.py           # 音频编解码
│   │   │   ├── vad/silero.py      # Silero VAD 语音活动检测（ONNX）
│   │   │   ├── kws/sherpa.py      # Sherpa KWS 关键词唤醒
│   │   │   └── asr/sherpa.py      # Sherpa ASR 离线语音识别（SenseVoice）
│   │   ├── tts/doubao.py          # 豆包 TTS 客户端（火山引擎）
│   │   └── protocols/
│   │       ├── websocket_protocol.py  # 小智 WebSocket 协议实现
│   │       └── typing.py              # 协议类型定义
│   └── utils/
│       ├── logger.py              # 彩色日志（XiaozhiLogger 单例）
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

## 核心组件

### MainApp (core/app.py)

应用主控制器，单例模式，管理全部服务生命周期。

- `instance(enable_xiaozhi, enable_openclaw)` → 单例获取
- `run(enable_api_server)` → 启动各服务
- `set_device_state(state)` → 管理设备状态（IDLE / LISTENING / SPEAKING / CONNECTING）
- `send_text(text)` → 发送文本到小智
- `send_to_openclaw(text, wait_response)` → 发送消息到 OpenClaw（返回 run_id 或回复文本）
- `send_to_openclaw_and_play_reply(text, wait_response)` → 发送并 TTS 播放回复
- `schedule(callback)` → 主线程任务队列
- `shutdown()` → 优雅关闭

**边界约束**:
- `MainApp` 是业务主循环和设备状态的单一入口
- `device_state` 以 `MainApp` 为准，其他模块通过代理回写，不各自维护平行状态
- `MainApp.loop` 是业务协程的主调度循环

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

### XiaoZhi (core/xiaozhi.py)

小智 AI WebSocket 协议客户端，单例模式。

- `connect()` / `disconnect()` → 连接管理
- `send_audio(frames)` / `send_text(text)` → 发送音频 / 文本
- `send_start_listening(mode)` / `send_abort_speaking(reason)` → 协议命令
- 回调委托：`on_incoming_audio`, `on_incoming_json`, `on_network_error` 等

**边界约束**:
- 只负责协议收发，不负责唤醒策略和连续对话策略
- `session_id` 必须由服务端消息更新，不能长期使用空值发送控制消息

### OpenClawManager (core/openclaw.py)

OpenClaw 网关客户端，管理 WebSocket 连接、消息分发、自动重连、TTS 播放。

- `initialize_from_config(enabled)` → 从 config 初始化
- `connect()` → 建立连接（Ed25519 设备身份认证）
- `send(text, wait_response)` → 发送消息，返回 run_id 或回复文本，失败返回 None
- `send_and_play(text, wait_response)` → 发送并 TTS 播放回复
- `is_connected()` / `is_enabled()` → 状态查询

**内部机制**:
- Ed25519 设备身份认证（密钥存储在 `app/openclaw/identity/`）
- WebSocket ping/pong + tick 事件监控连接健康
- 指数退避重连（初始 1s，最大 60s）
- 请求 ID 映射 `_pending: dict[str, asyncio.Future]` 追踪响应
- TTS 播放：`tts_speaker` 为 `"xiaoai"` 时使用小爱原生 TTS，否则使用豆包 TTS（支持流式）

**连接参数限制**:
- `client.id`: 必须是 OpenClaw 预定义常量
- `client.mode`: 必须是预定义常量
- `session_key`: 只从 config.py 读取

### OpenClawConversationController (core/openclaw_conversation.py)

OpenClaw 连续对话控制器。唤醒词触发后进入独立的 VAD → ASR → OpenClaw → TTS 循环。

- `start()` → 进入对话模式
- `stop()` → 退出对话
- `is_active()` → 状态查询

**对话循环** (`_run_one_turn`):
1. VAD 检测语音开始（`_wait_for_speech`）
2. 录制完整语音（VAD 帧 hook）
3. SherpaASR 离线识别
4. 退出关键词检测
5. 发送到 OpenClaw
6. TTS 播放回复（阻塞等待完成）
7. 恢复监听

**回声防护两层机制**:
1. `stop_recording` → kill 远端 arecord → 麦克风物理静音（音箱上有效）
2. `_wait_for_silence` → VAD 检测环境安静（本地调试兜底）

**边界约束**:
- 使用独立 VAD Future，不与 WakeupSessionManager 冲突
- TTS 完全阻塞，播放完成后才继续监听

### WakeupSessionManager (core/wakeup_session.py)

小智唤醒会话状态机，协调 KWS → VAD → 小智/OpenClaw 的唤醒流程。

- `wakeup(text, source)` → 处理唤醒（调用 `before_wakeup` 钩子，路由到 XiaoZhi 或 OpenClaw）
- `wait_next_step(timeout)` → 异步等待状态变化（带待决状态缓冲）
- `update_step(step, step_data)` → 更新步骤
- 事件回调：`on_interrupt()`, `on_wakeup()`, `on_tts_start()`, `on_tts_end()`, `on_speech()`, `on_silence()`
- `on_interrupt()` → 小爱唤醒时：cancel OpenClaw task、停止设备音频播放、恢复录音通道、stop XiaoAI conversation

**路由规则**（`before_wakeup` 返回值）:
- `"xiaozhi"` → 走小智流程
- `"openclaw"` → 走 OpenClaw 连续对话
- `None` → 不处理（用户自行处理）

**边界约束**:
- 它是"小智唤醒会话状态机"，不是通用事件总线
- 只允许缓存 `on_speech` / `on_silence` 等外部探测信号
- 不要缓存 `on_wakeup` / `on_interrupt` 等控制步骤

### XiaoAIConversationController (core/xiaoai_conversation.py)

小爱自身的连续对话管理。

- `handle_text_command(text, speaker)` → 处理退出 / 连续对话关键词
- `handle_listening_timeout(speaker)` → 超时重试逻辑
- `handle_audio_player_instruction(header_name)` → 检测播放器指令退出
- `handle_playing_status(playing_status, speaker)` → TTS 完成后重新唤醒

**边界约束**:
- 小爱连续对话和小智唤醒 / 会话超时是两套独立机制
- 只有在"小爱连续对话确实激活"时才允许停止
- 小智超时退出时不应打印"小爱停止连续对话"日志

### SpeakerManager (core/services/speaker.py)

音箱硬件控制。

- `play(text, url, buffer, blocking, timeout)` → 播放文字 / URL / PCM 缓冲
- `wake_up(awake, silent)` → 唤醒 / 休眠小爱
- `abort_xiaoai()` → 中断小爱当前操作
- `ask_xiaoai(text, silent)` → 让小爱执行指令
- `run_shell(command, timeout)` → RPC shell

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

### 模式 2: 小智 AI
```bash
XIAOZHI_ENABLE=1 uv run main.py
```
- 启动 VAD + KWS，唤醒后连接小智 AI
- KWS 初始化失败应视为启动失败

### 模式 3: OpenClaw
```bash
OPENCLAW_ENABLED=1 uv run main.py
```
- 小爱指令拦截 → 转发到 OpenClaw → TTS 播放结果

### 模式 4: 小智 + OpenClaw（混合）
```bash
XIAOZHI_ENABLE=1 OPENCLAW_ENABLED=1 uv run main.py
```
- config.py `before_wakeup` 按唤醒词路由到小智或 OpenClaw 连续对话
- OpenClaw 连续对话：VAD → ASR → OpenClaw → TTS 循环
- 退出关键词：config `openclaw.exit_keywords`

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
- `set_app/get_app`, `set_xiaozhi/get_xiaozhi`, `set_xiaoai/get_xiaoai`
- `set_vad/get_vad`, `set_kws/get_kws`, `set_speaker/get_speaker`
- `set_audio_codec/get_audio_codec`, `set_speech_frames/get_speech_frames`

### 兼容约束
- `CLI` 环境变量不再作为功能开关，不要引入依赖 `CLI` 的运行时分支
- `XIAOZHI_ENABLE=0` 时必须允许跳过 KWS 初始化
- `scripts/start.sh` 在仅小爱模式下不应检查 `core/models/` 下的模型文件

## 测试

```bash
# 无音箱流式冒烟测试
python3 tests/test_tts_stream.py

# 比较长文本 mp3/pcm 流式时延
python3 tests/test_tts_latency.py --formats mp3,pcm --rounds 3 --repeat 8

# OpenClaw 连通性测试
python3 tests/test_openclaw_live_connectivity.py
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

**注意**：OpenClaw 对话中 TTS 播放时会 `stop_recording` 防止回声。如果在此期间触发中断（"小爱同学"），必须在中断处理中调用 `start_recording` 恢复录音，否则 KWS 将因无音频数据而永久失效。

### on_interrupt 中断处理要点

`on_interrupt()` 触发时（用户喊"小爱同学"），需要完成以下步骤：
1. Cancel OpenClaw asyncio task
2. `killall tts_play.sh miplayer` — 停止阻塞 TTS
3. `mphelper pause` — 尝试停止非阻塞 TTS
4. `stop_playing` + `start_playing` — 重置 PCM 通道
5. `start_recording` — 恢复录音（KWS 依赖此通道）
6. `XiaoAI.stop_conversation()` — 停止连续对话

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
