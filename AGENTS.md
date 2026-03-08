# AGENTS.md - Bridge Module Guide

> 本目录 (`examples/bridge/`) 包含让小爱音箱接入小智 AI 和 OpenClaw 的完整实现。
> 通过接管音箱的音频输入输出，实现与第三方 AI 服务的对话。

## 项目架构

```
open-xiaoai/
├── examples/bridge/           # AI Bridge：小爱 + 小智 AI + OpenClaw
│   ├── main.py               # 程序入口
│   ├── config.py             # 用户配置文件（唤醒词、TTS、OpenClaw 等）
│   └── core/                 # 核心源码（原 xiaozhi/）
│       ├── app.py            # MainApp: 应用主控制器
│       ├── xiaoai.py         # XiaoAI: 小爱音箱接口（事件、TTS、控制）
│       ├── xiaozhi.py        # XiaoZhi: 小智 AI WebSocket 协议
│       ├── openclaw.py       # OpenClawManager: OpenClaw 网关连接
│       ├── ref.py            # 全局状态管理（get/set）
│       ├── event.py          # EventManager: 事件总线
│       ├── services/         # 服务层
│       │   ├── speaker.py    # SpeakerManager: 音箱控制（播放/TTS/唤醒）
│       │   ├── api_server.py # HTTP API 服务（远程控制）
│       │   ├── audio/        # 音频处理
│       │   │   ├── kws/      # 关键词唤醒（Sherpa）
│       │   │   ├── vad/      # 语音活动检测（Silero）
│       │   │   └── codec.py  # 音频编解码
│       │   └── protocols/    # 通信协议
│       └── utils/            # 工具类
├── packages/client-rust/     # 小爱音箱 Client 端补丁（Rust）
└── docs/                     # 文档
```

## 核心组件说明

### 1. MainApp (app.py)
应用主控制器，单例模式管理整个应用生命周期：
- 初始化 XiaoAI 服务（必须）
- 可选初始化 XiaoZhi（小智 AI）
- 可选初始化 OpenClaw
- 可选启动 API Server
- 管理音频设备状态（IDLE/LISTENING/SPEAKING/CONNECTING）

### 2. XiaoAI (xiaoai.py)
小爱音箱交互接口：
- `on_input_data`: 接收麦克风音频输入
- `on_output_data`: 发送音频到扬声器
- `on_event`: 处理小爱事件（语音识别结果、播放状态等）
- `SpeakerManager`: 控制播放、TTS、唤醒等

**重要**: 必须通过 `set_xiaoai(XiaoAI)` 注册，否则 `get_xiaoai()` 返回 None。

### 3. XiaoZhi (xiaozhi.py)
小智 AI 协议客户端：
- WebSocket 连接小智服务器
- 发送音频/文本，接收 AI 响应
- 支持 VAD（语音检测）和 KWS（关键词唤醒）

### 4. OpenClawManager (openclaw.py)
OpenClaw 网关客户端：
- WebSocket 连接到 OpenClaw Gateway
- `session_key`: 指定 OpenClaw session（默认 "main"）
- `send_message()`: 发送消息触发 AI 处理

**连接参数限制**:
- `client.id`: 必须是 OpenClaw 预定义常量（如 "gateway-client"）
- `client.mode`: 必须是预定义常量（如 "backend"）

### 5. SpeakerManager (services/speaker.py)
音箱控制接口：
- `play(text/url/buffer)`: 播放文字/链接/音频流
- `wake_up()`: 唤醒/休眠小爱
- `abort_xiaoai()`: 中断小爱当前操作
- `ask_xiaoai()`: 让小爱执行指令

### 6. 事件系统 (event.py)
- `EventManager.wakeup()`: 触发唤醒流程
- `before_wakeup`: 唤醒前回调（在 config.py 中配置）
- `after_wakeup`: 唤醒后回调

## 配置说明 (config.py)

```python
APP_CONFIG = {
    "wakeup": {
        "keywords": ["你好小智", "贾维斯"],  # 自定义唤醒词
        "timeout": 20,                          # 唤醒状态超时（秒）
        "before_wakeup": before_wakeup,         # 唤醒前回调
        "after_wakeup": after_wakeup,           # 退出唤醒回调
    },
    "vad": {
        "threshold": 0.10,      # 语音检测阈值
    },
    "xiaozhi": {
        "OTA_URL": "...",       # 小智 OTA 地址
        "WEBSOCKET_URL": "...", # 小智 WebSocket 地址
    },
    "openclaw": {
        "url": "ws://localhost:18789",
        "token": "",            # OpenClaw 认证令牌
        "session_key": "main",  # OpenClaw session 标识
    },
    "tts": {
        "doubao": {
            "app_id": "...",
            "access_key": "...",
        }
    }
}
```

## 开发规范

### 代码风格
- 使用中文注释和文档字符串
- 使用英文生成 commit message
- 类型提示: `dict[str, asyncio.Future]`

### 异步编程
- 所有 I/O 操作使用 `async/await`
- 线程安全使用 `asyncio.run_coroutine_threadsafe()`
- 全局事件循环在 `MainApp._run_event_loop()` 中运行

### 添加新功能
1. 在 `config.py` 中添加配置项
2. 在 `core/` 下创建模块
3. 在 `MainApp` 中初始化管理
4. 通过 `ref.py` 注册全局访问点

### 调试技巧
- 日志使用 `core.utils.logger.logger`
- 设置环境变量 `DEBUG=1` 开启详细日志
- API Server 提供 `/api/health` 健康检查端点

## 常见集成模式

### 模式 1: 仅小爱模式（默认）
```python
# 环境变量: XIAOZHI_ENABLE=0
MainApp.instance(enable_xiaozhi=False)
# 只启动小爱服务，无 AI 对话功能
```

### 模式 2: 小智 AI 模式
```python
# 环境变量: XIAOZHI_ENABLE=1
MainApp.instance(enable_xiaozhi=True)
# 启动 VAD + KWS，唤醒后连接小智 AI
```

### 模式 3: OpenClaw 代理模式
```python
# 环境变量: OPENCLAW_ENABLED=1
# 用户说"让龙虾 xxx" -> 转发到 OpenClaw -> 小爱播放结果
await app.send_to_openclaw("用户指令")
```

### 模式 4: 混合模式
```python
# XIAOZHI_ENABLE=1 + OPENCLAW_ENABLED=1
# - "你好小智" -> 唤醒小智 AI
# - "让龙虾 xxx" -> OpenClaw 代理
# - 其他指令 -> 小爱原生处理
```

## 重要依赖

- `open_xiaoai_server`: Rust 实现的底层服务（音频采集/播放）
- `sherpa-onnx`: 关键词唤醒
- `silero-vad`: 语音活动检测
- `websockets`: WebSocket 客户端
- `aiohttp`: HTTP API Server

## 测试命令

所有命令都在本目录 (`examples/bridge/`) 下执行：

```bash
# 运行仅小爱模式
python main.py

# 运行小智 AI 模式
XIAOZHI_ENABLE=1 python main.py

# 运行带 API Server
API_SERVER_ENABLE=1 XIAOZHI_ENABLE=1 python main.py

# 启用 OpenClaw
export OPENCLAW_ENABLED=1
export OPENCLAW_TOKEN="your_token"
python main.py
```

## 参考资源

- 项目主页: https://github.com/idootop/open-xiaoai
- 刷机教程: ../../docs/flash.md（从仓库根目录）
- Client 端: ../../packages/client-rust/README.md（从仓库根目录）
- 仓库结构: 本模块位于 `examples/bridge/`，Client 端补丁位于 `packages/client-rust/`
