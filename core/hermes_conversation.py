"""Hermes Agent 连续对话控制器。"""

import asyncio

import open_xiaoai_server

from core.hermes import HermesManager
from core.hermes_progress import HermesProgressNarrator
from core.streaming_conversation import StreamingConversationController
from core.utils.logger import logger


class HermesConversationController(StreamingConversationController):
    """支持结构化进度和流式最终语音的 Hermes 会话。"""

    CONFIG_PREFIX = "hermes"
    BACKEND_NAME = "Hermes"
    LOG_MODULE = "Hermes Conv"
    WAKEUP_SOURCE = "hermes"
    MANAGER = HermesManager
    STREAMING_DEFAULT = True

    def stop(self):
        """停止连续对话，并取消 complete 模式正在等待的 HTTP 请求。"""

        request_task = getattr(self, "_complete_request_task", None)
        super().stop()
        if request_task and not request_task.done():
            request_task.get_loop().call_soon_threadsafe(
                request_task.cancel
            )

    def _streaming_enabled(self) -> bool:
        """response_mode 是唯一的响应交付模式选择。"""

        return self.backend.get_response_mode() == "streaming"

    async def _request_backend_turn(
        self,
        text: str,
        *,
        play_send_sound: bool,
    ) -> tuple[str | None, bool]:
        if self._streaming_enabled():
            return await self._request_streaming_turn(
                text,
                play_send_sound=play_send_sound,
            )
        return await self._request_complete_turn(
            text,
            play_send_sound=play_send_sound,
        )

    async def _request_complete_turn(
        self,
        text: str,
        *,
        play_send_sound: bool,
    ) -> tuple[str | None, bool]:
        """发起一次明确的 non-streaming 请求并返回完整正文。"""

        prompt = text
        if self.backend._rule_prompt:
            prompt = text + "\n" + self.backend._rule_prompt

        request_task = asyncio.create_task(
            self.backend._request_chat_completion(prompt)
        )
        self._complete_request_task = request_task
        try:
            if play_send_sound:
                await self._play_send_sound()
            return await request_task, False
        except BaseException:
            if not request_task.done():
                request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)
            raise
        finally:
            if (
                getattr(self, "_complete_request_task", None)
                is request_task
            ):
                self._complete_request_task = None

    def _progress_config(self):
        value = self._cfg("progress", {})
        return value if isinstance(value, dict) else {}

    def _create_progress_narrator(self, text: str):
        return HermesProgressNarrator(text)

    async def _request_stream(
        self,
        prompt: str,
        *,
        on_delta,
        on_progress,
    ) -> str | None:
        return await self.backend.request_streaming_chat_completion(
            prompt,
            on_delta=on_delta,
            on_tool_progress=on_progress,
        )

    async def _play_tts(self, text: str):
        """等待一条语音的 Router 调用结束，传播其未处理的异常和取消。"""

        self._playback_token = open_xiaoai_server.begin_playback_session()
        try:
            # 上游 Router 返回 None，不是播放成功回执，也不能判为失败。
            # provider 回退由 Router 负责，此处不重播或重新请求 Agent。
            await self.backend._play_response_with_tts(
                text,
                tts_speaker=self.backend.get_tts_speaker_for_session_key(),
                playback_token=self._playback_token,
            )
        except Exception as exc:
            logger.error(
                f"TTS playback error: {type(exc).__name__}: {exc}",
                module=self.LOG_MODULE,
            )
            raise
        finally:
            self._playback_token = None
