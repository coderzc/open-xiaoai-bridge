"""外部对话后端共用的流式文本交付逻辑。"""

from __future__ import annotations

import asyncio
from typing import Any

from core.external_conversation import ExternalConversationController
from core.openai_stream import SentenceChunker
from core.ref import get_speaker
from core.speech_queue import SequentialSpeechQueue
from core.utils.logger import logger


class _SpeechWorkerFailed(RuntimeError):
    """Playback worker failed independently of the network stream."""


class StreamingConversationController(ExternalConversationController):
    """通过单个竞态安全的 TTS 工作线程交付流式回答。"""

    STREAMING_DEFAULT = False

    def stop(self):
        """停止会话，并取消当前由本 controller 持有的流式 turn。"""

        turn_task = getattr(self, "_streaming_turn_task", None)
        super().stop()
        if turn_task and not turn_task.done():
            turn_task.get_loop().call_soon_threadsafe(turn_task.cancel)

    def _stream_config(self) -> dict[str, Any]:
        value = self._cfg("streaming", self.STREAMING_DEFAULT)
        if isinstance(value, dict):
            return value
        return {"enabled": bool(value)}

    def _streaming_enabled(self) -> bool:
        return bool(
            self._stream_config().get(
                "enabled",
                self.STREAMING_DEFAULT,
            )
        )

    def _progress_config(self) -> dict[str, Any]:
        return {}

    def _create_progress_narrator(self, _text: str):
        return None

    async def _request_stream(
        self,
        prompt: str,
        *,
        on_delta,
        on_progress,
    ) -> str | None:
        """请求后端流；具体传输由子类定义。"""

        raise NotImplementedError

    async def _request_backend_turn(
        self,
        text: str,
        *,
        play_send_sound: bool,
    ) -> tuple[str | None, bool]:
        if not self._streaming_enabled():
            return await super()._request_backend_turn(
                text,
                play_send_sound=play_send_sound,
            )
        return await self._request_streaming_turn(
            text,
            play_send_sound=play_send_sound,
        )

    async def _request_streaming_turn(
        self,
        text: str,
        *,
        play_send_sound: bool,
    ) -> tuple[str | None, bool]:
        """将完整句段流式送入一个顺序 TTS 队列。"""

        prompt = text
        if self.backend._rule_prompt:
            prompt = text + "\n" + self.backend._rule_prompt

        stream_config = self._stream_config()
        chunker = SentenceChunker(
            min_chars=int(
                stream_config.get("sentence_min_chars", 24)
            ),
            max_chars=int(
                stream_config.get("sentence_max_chars", 160)
            ),
        )
        speech_queue = SequentialSpeechQueue()
        playback_ready = asyncio.Event()
        progress_stop = asyncio.Event()
        received_parts: list[str] = []
        delivered_final_chunks: list[str] = []
        narrator = self._create_progress_narrator(text)

        async def playback_worker():
            await playback_ready.wait()
            while True:
                item = await speech_queue.get()
                if item is None:
                    return
                try:
                    await self._play_tts(item.text)
                finally:
                    await speech_queue.finish(item)

        async def on_delta(delta: str):
            if not delta:
                return
            received_parts.append(delta)
            removed = await speech_queue.mark_final_started()
            if removed:
                logger.debug(
                    f"Skipped {removed} stale queued progress item(s)",
                    module=self.LOG_MODULE,
                )
            for sentence in chunker.feed(delta):
                delivered_final_chunks.append(sentence)
                await speech_queue.put_final(sentence)

        async def on_progress(event):
            if narrator is not None:
                narrator.observe(event)

        async def progress_announcer():
            progress = self._progress_config()
            if narrator is None or not bool(
                progress.get("enabled", True)
            ):
                return
            initial_delay = max(
                1.0,
                float(progress.get("initial_delay", 8)),
            )
            min_interval = max(
                3.0,
                float(progress.get("min_interval", 20)),
            )
            max_messages = max(
                0,
                int(progress.get("max_messages", 2)),
            )

            await playback_ready.wait()
            try:
                await asyncio.wait_for(
                    progress_stop.wait(),
                    timeout=initial_delay,
                )
                return
            except asyncio.TimeoutError:
                pass

            for _ in range(max_messages):
                if speech_queue.final_started or progress_stop.is_set():
                    return
                announcement = narrator.next_message()
                if announcement:
                    message, key = announcement
                    await speech_queue.put_progress(message, key=key)
                try:
                    await asyncio.wait_for(
                        progress_stop.wait(),
                        timeout=min_interval,
                    )
                    return
                except asyncio.TimeoutError:
                    continue

        worker_task: asyncio.Task | None = None
        progress_task: asyncio.Task | None = None
        stream_task: asyncio.Task | None = None
        turn_task = asyncio.current_task()
        self._streaming_turn_task = turn_task

        async def abort_turn():
            """丢弃旧语音并回收本 turn 创建的全部任务。"""

            progress_stop.set()
            removed = await speech_queue.abort()
            if removed:
                logger.debug(
                    f"Discarded {removed} queued speech item(s) on abort",
                    module=self.LOG_MODULE,
                )

            child_tasks = tuple(
                task
                for task in (stream_task, progress_task, worker_task)
                if task is not None
            )
            for task in child_tasks:
                if not task.done():
                    task.cancel()

            speaker = get_speaker()
            if speaker:
                try:
                    await speaker.stop_device_audio()
                except Exception as exc:
                    logger.debug(
                        f"Failed to stop aborted turn audio: {exc}",
                        module=self.LOG_MODULE,
                    )

            if child_tasks:
                await asyncio.gather(
                    *child_tasks,
                    return_exceptions=True,
                )
            await self._start_recording()

        async def cleanup_despite_cancellation():
            """重复 cancel 时仍等待 cleanup 真正结束。"""

            cleanup_task = asyncio.create_task(abort_turn())
            while not cleanup_task.done():
                try:
                    await asyncio.shield(cleanup_task)
                except asyncio.CancelledError:
                    continue
            await cleanup_task

        async def wait_for_stream():
            """等待流，同时让 playback worker 的异常能及时终止 turn。"""

            done, _pending = await asyncio.wait(
                {stream_task, worker_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if worker_task in done:
                try:
                    await worker_task
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    raise _SpeechWorkerFailed(
                        "Speech worker failed during streaming"
                    ) from exc
                if not stream_task.done():
                    raise _SpeechWorkerFailed(
                        "Speech worker exited before streaming completed"
                    )
            return await stream_task

        try:
            await self._stop_recording()
            # 从第一个子任务创建开始，全部生命周期都受本 try/except 保护。
            worker_task = asyncio.create_task(playback_worker())
            progress_task = asyncio.create_task(progress_announcer())
            logger.user_speech(
                prompt,
                module=f"{self.BACKEND_NAME}({self.backend._session_key})",
            )
            stream_task = asyncio.create_task(
                self._request_stream(
                    prompt,
                    on_delta=on_delta,
                    on_progress=on_progress,
                )
            )

            if play_send_sound:
                await self._play_send_sound()
            playback_ready.set()

            response: str | None = None
            try:
                response = await wait_for_stream()
            except asyncio.CancelledError:
                raise
            except _SpeechWorkerFailed:
                raise
            except Exception as exc:
                logger.error(
                    "Streaming interrupted; the Agent turn will not be "
                    f"submitted again: {type(exc).__name__}: {exc}",
                    module=self.LOG_MODULE,
                )
                # 交付已经收到但尚未成句的正文；stream helper 只有完整成功
                # 才写 history，因此这里不会把不完整回答记成成功轮次。
                for sentence in chunker.feed("", final=True):
                    delivered_final_chunks.append(sentence)
                    await speech_queue.put_final(sentence)
                interruption = "回答传输中断了，请再问一次"
                delivered_final_chunks.append(interruption)
                await speech_queue.put_final(interruption)
                response = "".join(received_parts).strip() or None

            # 成功流的末尾可能还留有一个不足最小句长的正文片段。
            for sentence in chunker.feed("", final=True):
                delivered_final_chunks.append(sentence)
                await speech_queue.put_final(sentence)

            progress_stop.set()
            await progress_task
            await speech_queue.drain()
            await worker_task
            return response, bool(delivered_final_chunks)
        except asyncio.CancelledError:
            await cleanup_despite_cancellation()
            raise
        except BaseException:
            await cleanup_despite_cancellation()
            raise
        finally:
            if getattr(self, "_streaming_turn_task", None) is turn_task:
                self._streaming_turn_task = None
