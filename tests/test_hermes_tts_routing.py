"""Hermes -> Router -> Speaker 链路回归；只模拟网络、设备与 Rust I/O。"""

import asyncio
import importlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch


sys.modules.setdefault("open_xiaoai_server", types.SimpleNamespace())


class _Config:
    def __init__(self):
        self.hermes = {
            "input_mode": "local_asr",
            "streaming": {"sentence_min_chars": 4, "sentence_max_chars": 80},
            "progress": {"enabled": False},
        }

    def get_app_config(self, path, default=None):
        value = {"hermes": self.hermes}
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value


class HermesTTSRoutingTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.controller_module = importlib.import_module("core.hermes_conversation")
        self.router_module = importlib.import_module("core.services.tts.router")
        self.speaker_module = importlib.import_module("core.services.speaker")
        self.external_module = importlib.import_module("core.external_conversation")
        self.streaming_module = importlib.import_module("core.streaming_conversation")
        hermes = importlib.import_module("core.hermes").HermesManager

        # 子类隔离测试状态，但不替换真实的 Hermes/Router/Controller 实现。
        class Backend(hermes):
            _tts_provider = "openai"
            _tts_speaker = "xiaoai"
            _tts_speed = 1.0
            _initialized = True
            _session_tts_speakers = {}
            _session_key = "test-speaker"
            _response_mode = "streaming"
            _rule_prompt = ""

        self.backend = Backend
        self.token = 0
        self.played_files = []
        self.controller = self.controller_module.HermesConversationController.__new__(
            self.controller_module.HermesConversationController
        )
        self.controller.backend = Backend
        self.controller.config = _Config()
        self.controller.active = True
        self.controller._playback_token = None
        self.controller._vad_future = None
        self.controller._xiaoai_asr_future = None
        self.controller._loop = None
        for name in ("_stop_recording", "_start_recording", "_play_send_sound", "_play_notify"):
            setattr(self.controller, name, AsyncMock())

        def begin():
            self.token += 1
            return self.token

        def stop(token):
            if token == self.token:
                self.token += 1

        async def play_file(path, *, sample_rate):
            self.assertTrue(Path(path).is_file())
            self.assertEqual(24000, sample_rate)
            # 上游文件接口独立创建 token，没有接收 controller token 的参数。
            self.native.begin_playback_session()
            self.played_files.append(path)
            # 和 Rust 既有契约一致：正常返回 None。

        self.native = types.SimpleNamespace(
            begin_playback_session=Mock(side_effect=begin),
            stop_tts_playback=Mock(side_effect=stop),
            play_audio_file=AsyncMock(side_effect=play_file),
            tts_play=AsyncMock(return_value=None),
            tts_stream_play=AsyncMock(return_value=None),
        )
        for module in (self.controller_module, self.router_module, self.speaker_module, self.external_module):
            self.enterContext(patch.object(module, "open_xiaoai_server", self.native))
        self.speaker = self.speaker_module.SpeakerManager.__new__(self.speaker_module.SpeakerManager)
        self.speaker.play = AsyncMock(return_value=True)
        self.speaker.stop_device_audio = AsyncMock()
        self.enterContext(patch.object(importlib.import_module("core.ref"), "get_speaker", return_value=self.speaker))
        self.enterContext(patch.object(self.streaming_module, "get_speaker", return_value=self.speaker))
        self.tts_config = self.enterContext(patch.object(
            self.router_module.ConfigManager.instance(), "get_app_config",
            return_value={
                "base_url": "http://tts.test/v1", "response_format": "wav",
                "app_id": "test-app", "access_key": "test-key",
            },
        ))
        self.synthesize = self.enterContext(patch.object(
            self.router_module.OpenAITTS, "synthesize", new=AsyncMock(return_value=b"audio"),
        ))

    async def turn(self, request):
        self.backend.request_streaming_chat_completion = AsyncMock(side_effect=request)
        return await self.controller._request_backend_turn("问题", play_send_sound=False)

    async def test_complete_uses_one_request_one_synthesis_and_upstream_file_api(self):
        self.backend._response_mode = "complete"
        self.backend._request_chat_completion = AsyncMock(return_value="完整回答")
        response, played = await self.controller._request_backend_turn("问题", play_send_sound=False)
        self.assertFalse(played)
        await self.controller._play_tts(response)
        self.backend._request_chat_completion.assert_awaited_once_with("问题")
        self.synthesize.assert_awaited_once_with("完整回答")
        self.assertEqual(2, self.native.begin_playback_session.call_count)
        self.native.play_audio_file.assert_awaited_once_with(
            self.played_files[0], sample_rate=24000,
        )
        self.assertIsNone(self.controller._playback_token)
        self.assertFalse(Path(self.played_files[0]).exists())
        self.speaker.play.assert_not_awaited()

    async def test_all_providers_follow_router_none_contract(self):
        for provider in ("xiaoai", "doubao", "openai", "mlx_audio"):
            with self.subTest(provider=provider):
                self.backend._tts_provider = provider
                self.assertIsNone(await self.controller._play_tts("逐句回答"))
                self.assertIsNone(self.controller._playback_token)
        self.speaker.play.assert_awaited_once_with(text="逐句回答", blocking=True)
        self.native.tts_play.assert_awaited_once()
        self.assertIsNotNone(self.native.tts_play.await_args.kwargs["playback_token"])
        self.assertEqual(2, self.native.play_audio_file.await_count)
        self.assertTrue(all(not Path(path).exists() for path in self.played_files))

    async def test_stream_waits_for_first_file_before_next_sentence(self):
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def play_file(path, *, sample_rate):
            self.played_files.append(path)
            self.assertTrue(Path(path).is_file())
            if len(self.played_files) == 1:
                first_started.set()
                await release_first.wait()
                self.assertTrue(Path(path).is_file())

        async def request(_text, *, on_delta, on_tool_progress):
            await on_delta("第一句话准备好了。第二句话也准备好了。")
            return "第一句话准备好了。第二句话也准备好了。"

        self.native.play_audio_file.side_effect = play_file
        task = asyncio.create_task(self.turn(request))
        await asyncio.wait_for(first_started.wait(), 1)
        self.assertEqual(1, self.synthesize.await_count)
        self.assertFalse(task.done())
        release_first.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(
            ["第一句话准备好了。", "第二句话也准备好了。"],
            [call.args[0] for call in self.synthesize.await_args_list],
        )
        self.assertTrue(all(not Path(path).exists() for path in self.played_files))
        self.backend.request_streaming_chat_completion.assert_awaited_once()

    async def test_stream_waits_for_native_fallback_before_next_sentence(self):
        entered_fallback = asyncio.Event()
        release_fallback = asyncio.Event()
        async def synthesize(text):
            if text.startswith("第一"):
                raise RuntimeError("provider failed")
            return b"audio"
        async def fallback(**_kwargs):
            entered_fallback.set()
            await release_fallback.wait()
            return True
        async def request(_text, *, on_delta, on_tool_progress):
            await on_delta("第一句话准备好了。第二句话也准备好了。")
            return "第一句话准备好了。第二句话也准备好了。"
        self.synthesize.side_effect = synthesize
        self.speaker.play.side_effect = fallback
        task = asyncio.create_task(self.turn(request))
        await asyncio.wait_for(entered_fallback.wait(), 1)
        self.assertEqual(1, self.synthesize.await_count)
        self.assertFalse(task.done())
        release_fallback.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(2, self.synthesize.await_count)
        self.speaker.play.assert_awaited_once()
        self.native.play_audio_file.assert_awaited_once()

    async def test_playing_progress_finishes_before_final_through_router(self):
        self.controller.config.hermes["progress"] = {
            "enabled": True, "initial_delay": 1, "max_messages": 1,
        }
        progress_started = asyncio.Event()
        release_progress = asyncio.Event()
        final_queued = asyncio.Event()
        async def play_file(path, **_kwargs):
            self.played_files.append(path)
            if len(self.played_files) == 1:
                progress_started.set()
                await release_progress.wait()
        async def request(_text, *, on_delta, on_tool_progress):
            progress = importlib.import_module("core.hermes_progress").HermesToolProgress
            await on_tool_progress(progress("web_search", "running", "call-1"))
            await progress_started.wait()
            await on_delta("最终回答已经准备好了。")
            final_queued.set()
            return "最终回答已经准备好了。"
        self.native.play_audio_file.side_effect = play_file
        task = asyncio.create_task(self.turn(request))
        await asyncio.wait_for(final_queued.wait(), 2)
        self.assertEqual(1, self.synthesize.await_count)
        release_progress.set()
        await asyncio.wait_for(task, 1)
        self.assertEqual(
            ["正在查找相关资料", "最终回答已经准备好了。"],
            [call.args[0] for call in self.synthesize.await_args_list],
        )
        self.assertTrue(all(not Path(path).exists() for path in self.played_files))

    async def test_complete_task_cancellation_during_synthesis_or_file_does_not_fallback(self):
        for phase in ("synthesis", "file"):
            with self.subTest(phase=phase):
                self.controller.active = True
                self.backend._response_mode = "complete"
                started = asyncio.Event()
                release = asyncio.Event()
                target = self.synthesize if phase == "synthesis" else self.native.play_audio_file
                original_effect = target.side_effect
                async def delayed(*_args, **_kwargs):
                    started.set()
                    await release.wait()
                    return b"audio" if phase == "synthesis" else None
                target.side_effect = delayed
                task = asyncio.create_task(self.controller._play_tts("旧回答"))
                await asyncio.wait_for(started.wait(), 1)
                self.controller.stop()
                # 完整唤醒中断还会取消会话任务；单独停止旧 token 不是文件取消保证。
                task.cancel()
                replacement = self.native.begin_playback_session()
                with self.assertRaises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 1)
                self.assertEqual(replacement, self.token)
                self.speaker.play.assert_not_awaited()
                if phase == "file":
                    path = self.native.play_audio_file.await_args.args[0]
                    self.assertFalse(Path(path).exists())
                target.side_effect = original_effect

    async def test_stream_stop_discards_remaining_sentences_and_cleans_file(self):
        started = asyncio.Event()
        async def play_file(path, **_kwargs):
            self.played_files.append(path)
            started.set()
            await asyncio.Future()
        async def request(_text, *, on_delta, on_tool_progress):
            await on_delta("第一句准备好了。第二句准备好了。第三句准备好了。")
            await asyncio.Future()
        self.native.play_audio_file.side_effect = play_file
        task = asyncio.create_task(self.turn(request))
        await asyncio.wait_for(started.wait(), 1)
        self.controller.stop()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertEqual(1, self.synthesize.await_count)
        self.assertFalse(Path(self.played_files[0]).exists())
        self.speaker.play.assert_not_awaited()
        self.speaker.stop_device_audio.assert_awaited_once()
        self.controller._start_recording.assert_awaited_once()
        self.backend.request_streaming_chat_completion.assert_awaited_once()

    async def test_router_handled_failure_is_not_a_result_or_second_fallback(self):
        for failure in (False, RuntimeError("fallback failed")):
            with self.subTest(failure=failure):
                self.synthesize.reset_mock()
                self.speaker.play.reset_mock()
                self.synthesize.side_effect = RuntimeError("provider failed")
                self.speaker.play.side_effect = failure if isinstance(failure, Exception) else None
                self.speaker.play.return_value = failure
                self.assertIsNone(await self.controller._play_tts("无法播放"))
                self.synthesize.assert_awaited_once()
                self.speaker.play.assert_awaited_once()
                self.assertIsNone(self.controller._playback_token)

    async def test_stream_handled_failure_continues_without_resubmitting_agent(self):
        async def request(_text, *, on_delta, on_tool_progress):
            await on_delta("第一句话准备好了。第二句话也准备好了。")
            return "第一句话准备好了。第二句话也准备好了。"

        self.synthesize.side_effect = [RuntimeError("provider failed"), b"audio"]
        self.speaker.play.return_value = False
        response, dispatched = await self.turn(request)
        self.assertTrue(dispatched)  # 表示不再完整重播，不是实体出声确认。
        self.assertEqual("第一句话准备好了。第二句话也准备好了。", response)
        self.assertEqual(2, self.synthesize.await_count)
        self.speaker.play.assert_awaited_once()
        self.native.play_audio_file.assert_awaited_once()
        self.backend.request_streaming_chat_completion.assert_awaited_once()

    async def test_complete_recovers_listening_after_router_none(self):
        self.backend._response_mode = "complete"
        self.backend._tts_provider = "xiaoai"
        self.backend._request_chat_completion = AsyncMock(return_value="完整回答")
        self.controller._wait_for_xiaoai_asr_text = AsyncMock(return_value="问题")
        self.assertEqual("continue", await self.controller._run_one_turn_with_xiaoai_asr())
        self.controller._stop_recording.assert_awaited_once()
        self.controller._play_notify.assert_awaited_once()
        self.controller._start_recording.assert_awaited_once()
        self.speaker.play.assert_awaited_once_with(text="完整回答", blocking=True)

    async def test_speaker_file_call_keeps_upstream_signature(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "audio.wav"
            path.touch()
            self.native.play_audio_file.side_effect = None
            self.native.play_audio_file.return_value = None
            self.assertTrue(await self.speaker.play_server_file(str(path)))
            self.native.play_audio_file.assert_awaited_once_with(str(path), sample_rate=24000)
            self.assertEqual(1, self.native.play_audio_file.await_count)


if __name__ == "__main__":
    unittest.main()
