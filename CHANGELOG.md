# Changelog

All notable changes to this project will be documented in this file.

## v1.0.5 - 2026-03-29

### 重点更新

- 新增小爱原生 ASR 模式 (`OPENCLAW_XIAOAI_NATIVE_ASR`)，可在 OpenClaw 连续对话中使用小爱自带的语音识别能力，降低对离线 ASR 模型的依赖。
- 新增可配置的音频输入增益 (`AUDIO_INPUT_GAIN`)，支持调节麦克风输入音量以优化唤醒词识别灵敏度。
- 新增音频输入开关 (`AUDIO_INPUT_ENABLE`)，可在不需要音频输入时禁用以节省系统资源。
- 新增发送消息提示音，改善 OpenClaw 连续对话的交互体验。(#11 by @codertinat)

### 修复与优化

- 修复 OpenClaw 小爱原生 ASR 模式下的超时处理，确保桥接超时配置被正确遵循。
- 优化环境变量命名：`OPENCLAW_ENABLED` → `OPENCLAW_ENABLE`（保留向后兼容，新变量优先）。
- 优化 CMake 启动脚本，修复构建相关问题。(#11 by @codertinat)

### 文档更新

- 补充音频输入增益配置的 FAQ 说明。
- 优化 Docker FAQ 格式，统一文档风格。
- 更新 OpenClaw 连接说明文档。

### Full Changelog

- https://github.com/coderzc/open-xiaoai-bridge/compare/v1.0.4...v1.0.5

## v1.0.4 - 2026-03-26

### 重点更新

- 新增可配置的 ASR 后端，支持通过配置切换不同语音识别模型。
- 优化设备端音频播放链路，通过延迟启动播放降低 `aplay` underrun 问题。
- 优化长时间运行场景下的内部状态管理，减少潜在内存泄漏风险。

### 修复与优化

- 修复 `after_wakeup` 回调中未正确透传 `source` 参数的问题，改善小智/OpenClaw 会话退出后的收尾逻辑。
- 调整 XiaoZhi、XiaoAI、OpenClaw 以及原生音频相关实现，优化稳定性与部分边界行为。
- 补充和整理 Docker / README 相关说明，提升部署与使用时的可读性。

### 文档更新

- 补充并整理项目文档说明，优化 README 的来源说明、致谢与相关文案表达。
- 更新 LICENSE 中的版权声明，保留上游作者信息并补充当前项目维护者信息。
- 更新 Docker 使用说明，改善 Windows 用户的部署体验。(#8 by @JackieQiang)

### Full Changelog

- https://github.com/coderzc/open-xiaoai-bridge/compare/v1.0.3...v1.0.4

## v1.0.3 - 2026-03-25

### 重点更新

- 豆包 TTS 升级支持新的 2.0 音色，并补充配套的辅助脚本与接口文档，便于查询和验证可用音色。
- 新增 `scripts/clone_voice.py` 声音复刻脚本，支持提交音频样本并查询训练状态。
- 新增 `scripts/generate_tts.py` 音频生成脚本，可按指定 `speaker_id`、文本和情感参数导出音频文件。
- 新增播放服务端音频文件的能力，可通过 API 直接下发本地文件进行播放。
- 优化 OpenClaw TTS 打断与设备音频关闭流程，减少播放被打断后残留音频状态未清理的问题。

### 修复与优化

- 修复外部唤醒词触发时，小爱仍然回声式回复的问题，降低路由到第三方 AI 时的干扰。
- 修复用户喊出“小爱同学”打断后，小智唤醒会话没有完全恢复的问题，避免后续唤醒失效。
- 在 Doubao TTS API 返回成功前增加请求校验，避免无效请求被误判为成功。
- 优化 Doubao TTS 的错误处理与日志输出，减少重复报错，并在流式/后台播放失败时保留更完整的上下文。
- 调整 `docker-compose.yml`，移除 `network_mode: host`，改善默认 Docker Compose 部署的兼容性。
- 调整部分 XiaoZhi/OpenClaw 内部流程与日志细节，减少连续对话等待和排障成本。

### 文档更新

- 补充 Doubao TTS 接口、声音复刻和指定音色导出脚本的使用说明。

### Full Changelog

- https://github.com/coderzc/open-xiaoai-bridge/compare/v1.0.2...v1.0.3
