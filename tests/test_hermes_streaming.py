import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class _Config:
    def __init__(self, values):
        self.values = values

    def get_app_config(self, path, default=None):
        value = self.values
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value


class HermesStreamingTest(unittest.TestCase):
    def setUp(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        server = sys.modules.get("open_xiaoai_server")
        if server is None:
            server = types.SimpleNamespace()
            sys.modules["open_xiaoai_server"] = server
        if not hasattr(server, "decode_audio"):
            server.decode_audio = lambda *_args, **_kwargs: b""
        if not hasattr(server, "begin_playback_session"):
            server.begin_playback_session = lambda: 1

        module = importlib.import_module("core.hermes_conversation")
        self.module = module
        self.streaming_module = importlib.import_module(
            "core.streaming_conversation"
        )
        self.controller = module.HermesConversationController.__new__(
            module.HermesConversationController
        )
        self.controller.config = _Config(
            {
                "hermes": {
                    "streaming": {
                        "enabled": True,
                        "sentence_min_chars": 4,
                        "sentence_max_chars": 80,
                    },
                    "progress": {"enabled": False},
                    "input_mode": "xiaoai_asr",
                }
            }
        )
        self.controller._playback_token = None
        self.controller._vad_future = None
        self.controller._xiaoai_asr_future = None
        self.controller._loop = None
        self.controller._stop_recording = mock.AsyncMock()
        self.controller._start_recording = mock.AsyncMock()
        self.controller._play_send_sound = mock.AsyncMock()

    def test_start_does_not_rotate_hermes_conversation(self):
        self.controller.active = False
        self.controller.backend = types.SimpleNamespace(
            begin_conversation=mock.Mock()
        )

        async def scenario():
            with mock.patch.object(
                self.module.StreamingConversationController,
                "start",
                new=mock.AsyncMock(),
            ) as parent_start:
                await self.controller.start()
                return parent_start

        parent_start = asyncio.run(scenario())

        self.controller.backend.begin_conversation.assert_not_called()
        parent_start.assert_awaited_once_with()

    def test_interruption_notice_is_not_followed_by_generic_no_reply(self):
        self.controller._wait_for_xiaoai_asr_text = mock.AsyncMock(
            return_value="问题"
        )
        self.controller._request_backend_turn = mock.AsyncMock(
            return_value=(None, True)
        )
        self.controller._play_notify = mock.AsyncMock()
        speaker = mock.AsyncMock()

        external_module = importlib.import_module(
            "core.external_conversation"
        )
        with mock.patch.object(
            external_module,
            "get_speaker",
            return_value=speaker,
        ):
            result = asyncio.run(
                self.controller._run_one_turn_with_xiaoai_asr()
            )

        self.assertEqual("continue", result)
        speaker.play.assert_not_awaited()
        self.controller._play_notify.assert_awaited_once_with()
        self.controller._start_recording.assert_awaited_once_with()

    def test_final_sentence_starts_before_stream_finishes(self):
        async def scenario():
            first_spoken = asyncio.Event()
            release_first = asyncio.Event()
            stream_completed = False
            played = []

            async def play(text):
                played.append(text)
                if len(played) == 1:
                    first_spoken.set()
                    await release_first.wait()

            async def request(
                _text,
                *,
                on_delta,
                on_tool_progress,
            ):
                nonlocal stream_completed
                del on_tool_progress
                await on_delta("第一句已经好了。")
                await first_spoken.wait()
                self.assertFalse(stream_completed)
                release_first.set()
                await on_delta("第二句也好了。")
                stream_completed = True
                return "第一句已经好了。第二句也好了。"

            self.controller._play_tts = play
            self.controller.backend = types.SimpleNamespace(
                _rule_prompt="",
                _session_key="agent:hermes:test",
                request_streaming_chat_completion=request,
            )
            result = await self.controller._request_streaming_turn(
                "问题",
                play_send_sound=False,
            )
            return result, played

        result, played = asyncio.run(scenario())

        self.assertEqual(
            ("第一句已经好了。第二句也好了。", True),
            result,
        )
        self.assertEqual(
            ["第一句已经好了。", "第二句也好了。"],
            played,
        )

    def test_repeated_tool_events_create_one_rate_limited_progress(self):
        async def request(
            _text,
            *,
            on_delta,
            on_tool_progress,
        ):
            progress_module = importlib.import_module(
                "core.hermes_progress"
            )
            for call_id in ("call-1", "call-2", "call-3"):
                await on_tool_progress(
                    progress_module.HermesToolProgress(
                        tool="web_search",
                        status="running",
                        tool_call_id=call_id,
                        label="curl https://internal.example",
                    )
                )
            await asyncio.sleep(1.05)
            await on_delta("这是最终答案。")
            return "这是最终答案。"

        self.controller.config.values["hermes"]["progress"] = {
            "enabled": True,
            "initial_delay": 1,
            "min_interval": 20,
            "max_messages": 2,
        }
        played = []
        self.controller._play_tts = mock.AsyncMock(
            side_effect=lambda text: played.append(text)
        )
        self.controller.backend = types.SimpleNamespace(
            _rule_prompt="",
            _session_key="agent:hermes:test",
            request_streaming_chat_completion=request,
        )

        asyncio.run(
            self.controller._request_streaming_turn(
                "查一下最新资料",
                play_send_sound=False,
            )
        )

        self.assertEqual(
            ["正在查找相关资料", "这是最终答案。"],
            played,
        )
        self.assertNotIn("web_search", "".join(played))
        self.assertNotIn("http", "".join(played))

    def test_failure_before_spoken_text_does_not_resubmit_turn(self):
        async def request(
            _text,
            *,
            on_delta,
            on_tool_progress,
        ):
            del on_delta, on_tool_progress
            raise RuntimeError("stream unavailable")

        played = []
        self.controller._play_tts = lambda text: _append_async(
            played,
            text,
        )
        resubmit = mock.AsyncMock(return_value="不应执行的第二次回答。")
        self.controller.backend = types.SimpleNamespace(
            _rule_prompt="",
            _session_key="agent:hermes:test",
            request_streaming_chat_completion=request,
            _request_chat_completion=resubmit,
        )

        result = asyncio.run(
            self.controller._request_streaming_turn(
                "问题",
                play_send_sound=False,
            )
        )

        self.assertEqual((None, True), result)
        self.assertEqual(["回答传输中断了，请再问一次"], played)
        resubmit.assert_not_awaited()

    def test_failure_after_spoken_text_does_not_replay_answer(self):
        async def request(
            _text,
            *,
            on_delta,
            on_tool_progress,
        ):
            del on_tool_progress
            await on_delta("已经播出的第一句。")
            raise RuntimeError("connection lost")

        played = []
        self.controller._play_tts = lambda text: _append_async(
            played,
            text,
        )
        resubmit = mock.AsyncMock(return_value="不应重复的完整回答。")
        self.controller.backend = types.SimpleNamespace(
            _rule_prompt="",
            _session_key="agent:hermes:test",
            request_streaming_chat_completion=request,
            _request_chat_completion=resubmit,
        )

        result = asyncio.run(
            self.controller._request_streaming_turn(
                "问题",
                play_send_sound=False,
            )
        )

        self.assertEqual(("已经播出的第一句。", True), result)
        self.assertEqual(
            ["已经播出的第一句。", "回答传输中断了，请再问一次"],
            played,
        )
        resubmit.assert_not_awaited()

    def test_tool_progress_then_stream_failure_does_not_resubmit(self):
        async def request(
            _text,
            *,
            on_delta,
            on_tool_progress,
        ):
            del on_delta
            progress_module = importlib.import_module(
                "core.hermes_progress"
            )
            await on_tool_progress(
                progress_module.HermesToolProgress(
                    tool="web_search",
                    status="running",
                    tool_call_id="call-1",
                )
            )
            raise RuntimeError("connection lost after tool side effect")

        played = []
        self.controller._play_tts = lambda text: _append_async(
            played,
            text,
        )
        resubmit = mock.AsyncMock(return_value="不应执行")
        self.controller.backend = types.SimpleNamespace(
            _rule_prompt="",
            _session_key="agent:hermes:test",
            request_streaming_chat_completion=request,
            _request_chat_completion=resubmit,
        )

        result = asyncio.run(
            self.controller._request_streaming_turn(
                "执行任务",
                play_send_sound=False,
            )
        )

        self.assertEqual((None, True), result)
        self.assertEqual(["回答传输中断了，请再问一次"], played)
        resubmit.assert_not_awaited()

    def test_unfinished_delta_is_flushed_without_resubmitting(self):
        async def request(
            _text,
            *,
            on_delta,
            on_tool_progress,
        ):
            del on_tool_progress
            await on_delta("尚未成句的片段")
            raise RuntimeError("connection lost")

        played = []
        self.controller._play_tts = lambda text: _append_async(
            played,
            text,
        )
        resubmit = mock.AsyncMock(return_value="不应执行")
        self.controller.backend = types.SimpleNamespace(
            _rule_prompt="",
            _session_key="agent:hermes:test",
            request_streaming_chat_completion=request,
            _request_chat_completion=resubmit,
        )

        result = asyncio.run(
            self.controller._request_streaming_turn(
                "问题",
                play_send_sound=False,
            )
        )

        self.assertEqual(("尚未成句的片段", True), result)
        self.assertEqual(
            ["尚未成句的片段", "回答传输中断了，请再问一次"],
            played,
        )
        resubmit.assert_not_awaited()

    def test_abort_while_first_of_three_final_chunks_is_playing(self):
        async def scenario():
            first_started = asyncio.Event()
            first_cancelled = asyncio.Event()
            keep_stream_open = asyncio.Event()
            played = []
            speaker = mock.AsyncMock()

            async def play(text):
                played.append(text)
                if len(played) == 1:
                    first_started.set()
                    try:
                        await asyncio.Future()
                    except asyncio.CancelledError:
                        first_cancelled.set()
                        raise

            async def request(
                _text,
                *,
                on_delta,
                on_tool_progress,
            ):
                del on_tool_progress
                await on_delta("第一句已经就绪。第二句已经就绪。第三句已经就绪。")
                await keep_stream_open.wait()
                return "不应完成"

            self.controller.active = True
            self.controller._play_tts = play
            self.controller.backend = types.SimpleNamespace(
                _rule_prompt="",
                _session_key="agent:hermes:test",
                request_streaming_chat_completion=request,
            )
            baseline = set(asyncio.all_tasks())
            with mock.patch.object(
                self.streaming_module,
                "get_speaker",
                return_value=speaker,
            ):
                task = asyncio.create_task(
                    self.controller._request_streaming_turn(
                        "问题",
                        play_send_sound=False,
                    )
                )
                await first_started.wait()
                self.controller.stop()
                self.controller.stop()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(0)
                leftovers = [
                    item
                    for item in asyncio.all_tasks() - baseline
                    if not item.done()
                ]
            return played, first_cancelled.is_set(), leftovers, speaker

        played, first_cancelled, leftovers, speaker = asyncio.run(scenario())
        self.assertEqual(["第一句已经就绪。"], played)
        self.assertTrue(first_cancelled)
        self.assertEqual([], leftovers)
        speaker.stop_device_audio.assert_awaited_once()
        self.controller._start_recording.assert_awaited_once()

    def test_abort_while_send_sound_blocks_reclaims_all_children(self):
        async def scenario():
            send_sound_started = asyncio.Event()
            stream_started = asyncio.Event()
            stream_cancelled = asyncio.Event()
            speaker = mock.AsyncMock()

            async def send_sound():
                send_sound_started.set()
                await asyncio.Future()

            async def request(
                _text,
                *,
                on_delta,
                on_tool_progress,
            ):
                del on_delta, on_tool_progress
                stream_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    stream_cancelled.set()
                    raise

            self.controller._play_send_sound = send_sound
            self.controller._play_tts = mock.AsyncMock()
            self.controller.backend = types.SimpleNamespace(
                _rule_prompt="",
                _session_key="agent:hermes:test",
                request_streaming_chat_completion=request,
            )
            baseline = set(asyncio.all_tasks())
            with mock.patch.object(
                self.streaming_module,
                "get_speaker",
                return_value=speaker,
            ):
                task = asyncio.create_task(
                    self.controller._request_streaming_turn(
                        "问题",
                        play_send_sound=True,
                    )
                )
                await send_sound_started.wait()
                await stream_started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                await asyncio.sleep(0)
                leftovers = [
                    item
                    for item in asyncio.all_tasks() - baseline
                    if not item.done()
                ]
            return stream_cancelled.is_set(), leftovers, speaker

        stream_cancelled, leftovers, speaker = asyncio.run(scenario())
        self.assertTrue(stream_cancelled)
        self.assertEqual([], leftovers)
        speaker.stop_device_audio.assert_awaited_once()
        self.controller._start_recording.assert_awaited_once()

    def test_send_sound_exception_aborts_children_and_restores_recording(self):
        async def scenario():
            stream_cancelled = asyncio.Event()
            speaker = mock.AsyncMock()

            async def request(
                _text,
                *,
                on_delta,
                on_tool_progress,
            ):
                del on_delta, on_tool_progress
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    stream_cancelled.set()
                    raise

            self.controller._play_send_sound = mock.AsyncMock(
                side_effect=RuntimeError("beep failed")
            )
            self.controller._play_tts = mock.AsyncMock()
            self.controller.backend = types.SimpleNamespace(
                _rule_prompt="",
                _session_key="agent:hermes:test",
                request_streaming_chat_completion=request,
            )
            baseline = set(asyncio.all_tasks())
            with mock.patch.object(
                self.streaming_module,
                "get_speaker",
                return_value=speaker,
            ):
                with self.assertRaisesRegex(RuntimeError, "beep failed"):
                    await self.controller._request_streaming_turn(
                        "问题",
                        play_send_sound=True,
                    )
                await asyncio.sleep(0)
                leftovers = [
                    item
                    for item in asyncio.all_tasks() - baseline
                    if not item.done()
                ]
            return stream_cancelled.is_set(), leftovers, speaker

        stream_cancelled, leftovers, speaker = asyncio.run(scenario())
        self.assertTrue(stream_cancelled)
        self.assertEqual([], leftovers)
        speaker.stop_device_audio.assert_awaited_once()
        self.controller._start_recording.assert_awaited_once()

    def test_abort_unfinished_delta_never_plays_it(self):
        async def scenario():
            delta_received = asyncio.Event()
            speaker = mock.AsyncMock()
            played = []

            async def request(
                _text,
                *,
                on_delta,
                on_tool_progress,
            ):
                del on_tool_progress
                await on_delta("短片段")
                delta_received.set()
                await asyncio.Future()

            self.controller._play_tts = lambda text: _append_async(
                played,
                text,
            )
            self.controller.backend = types.SimpleNamespace(
                _rule_prompt="",
                _session_key="agent:hermes:test",
                request_streaming_chat_completion=request,
            )
            with mock.patch.object(
                self.streaming_module,
                "get_speaker",
                return_value=speaker,
            ):
                task = asyncio.create_task(
                    self.controller._request_streaming_turn(
                        "问题",
                        play_send_sound=False,
                    )
                )
                await delta_received.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            return played

        self.assertEqual([], asyncio.run(scenario()))

    def test_abort_progress_in_playback_never_continues_to_final(self):
        async def scenario():
            progress_started = asyncio.Event()
            progress_cancelled = asyncio.Event()
            speaker = mock.AsyncMock()
            original_wait_for = asyncio.wait_for
            initial_timeout_injected = False
            played = []

            async def controlled_wait_for(awaitable, timeout):
                nonlocal initial_timeout_injected
                if not initial_timeout_injected:
                    initial_timeout_injected = True
                    awaitable.close()
                    raise asyncio.TimeoutError
                return await original_wait_for(awaitable, timeout)

            async def play(text):
                played.append(text)
                progress_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    progress_cancelled.set()
                    raise

            async def request(
                _text,
                *,
                on_delta,
                on_tool_progress,
            ):
                del on_delta
                progress_module = importlib.import_module(
                    "core.hermes_progress"
                )
                await on_tool_progress(
                    progress_module.HermesToolProgress(
                        tool="web_search",
                        status="running",
                        tool_call_id="call-1",
                    )
                )
                await asyncio.Future()

            self.controller.config.values["hermes"]["progress"] = {
                "enabled": True,
                "initial_delay": 1,
                "min_interval": 20,
                "max_messages": 1,
            }
            self.controller._play_tts = play
            self.controller.backend = types.SimpleNamespace(
                _rule_prompt="",
                _session_key="agent:hermes:test",
                request_streaming_chat_completion=request,
            )
            with (
                mock.patch.object(
                    self.streaming_module,
                    "get_speaker",
                    return_value=speaker,
                ),
                mock.patch.object(
                    self.streaming_module.asyncio,
                    "wait_for",
                    side_effect=controlled_wait_for,
                ),
            ):
                task = asyncio.create_task(
                    self.controller._request_streaming_turn(
                        "查资料",
                        play_send_sound=False,
                    )
                )
                await progress_started.wait()
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            return played, progress_cancelled.is_set()

        played, progress_cancelled = asyncio.run(scenario())
        # 这个受控竞态在 tool event 到达前先触发 long-wait，
        # 因而只能使用“尚无真实工具事件”的通用提示。
        self.assertEqual(["还在为你处理，请稍等"], played)
        self.assertTrue(progress_cancelled)

    def test_old_aborted_turn_cannot_speak_after_new_turn_starts(self):
        async def scenario():
            old_started = asyncio.Event()
            speaker = mock.AsyncMock()
            played = []

            async def old_play(text):
                played.append(("old", text))
                old_started.set()
                await asyncio.Future()

            async def old_request(
                _text,
                *,
                on_delta,
                on_tool_progress,
            ):
                del on_tool_progress
                await on_delta("旧会话第一句。旧会话第二句。")
                await asyncio.Future()

            self.controller._play_tts = old_play
            self.controller.backend = types.SimpleNamespace(
                _rule_prompt="",
                _session_key="agent:hermes:old",
                request_streaming_chat_completion=old_request,
            )
            with mock.patch.object(
                self.streaming_module,
                "get_speaker",
                return_value=speaker,
            ):
                old_task = asyncio.create_task(
                    self.controller._request_streaming_turn(
                        "旧问题",
                        play_send_sound=False,
                    )
                )
                await old_started.wait()
                old_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await old_task

                new_controller = self.module.HermesConversationController.__new__(
                    self.module.HermesConversationController
                )
                new_controller.config = self.controller.config
                new_controller._playback_token = None
                new_controller._stop_recording = mock.AsyncMock()
                new_controller._start_recording = mock.AsyncMock()
                new_controller._play_send_sound = mock.AsyncMock()
                new_controller._play_tts = lambda text: _append_async(
                    played,
                    ("new", text),
                )

                async def new_request(
                    _text,
                    *,
                    on_delta,
                    on_tool_progress,
                ):
                    del on_tool_progress
                    await on_delta("新会话回答。")
                    return "新会话回答。"

                new_controller.backend = types.SimpleNamespace(
                    _rule_prompt="",
                    _session_key="agent:hermes:new",
                    request_streaming_chat_completion=new_request,
                )
                await new_controller._request_streaming_turn(
                    "新问题",
                    play_send_sound=False,
                )
                await asyncio.sleep(0)
            return played

        self.assertEqual(
            [
                ("old", "旧会话第一句。"),
                ("new", "新会话回答。"),
            ],
            asyncio.run(scenario()),
        )

    def test_mode_like_phrases_are_forwarded_unchanged(self):
        complete = mock.AsyncMock(return_value="回答")
        self.controller.backend = types.SimpleNamespace(
            _rule_prompt="",
            get_response_mode=lambda: "complete",
            _request_chat_completion=complete,
        )

        result = asyncio.run(
            self.controller._request_backend_turn(
                "详细查一下，不要简单说",
                play_send_sound=False,
            )
        )

        self.assertEqual(("回答", False), result)
        complete.assert_awaited_once_with(
            "详细查一下，不要简单说"
        )

    def test_streaming_mode_uses_sse_and_never_calls_complete(self):
        async def stream(
            _text,
            *,
            on_delta,
            on_tool_progress,
        ):
            del on_tool_progress
            await on_delta("流式回答完成。")
            return "流式回答完成。"

        complete = mock.AsyncMock(return_value="不应调用")
        played = []
        self.controller._play_tts = lambda text: _append_async(
            played,
            text,
        )
        self.controller.backend = types.SimpleNamespace(
            _rule_prompt="",
            _session_key="agent:hermes:test",
            get_response_mode=lambda: "streaming",
            request_streaming_chat_completion=stream,
            _request_chat_completion=complete,
        )

        result = asyncio.run(
            self.controller._request_backend_turn(
                "问题",
                play_send_sound=False,
            )
        )

        self.assertEqual(("流式回答完成。", True), result)
        self.assertEqual(["流式回答完成。"], played)
        complete.assert_not_awaited()

    def test_complete_mode_skips_stream_queue_and_plays_once(self):
        complete = mock.AsyncMock(return_value="完整回答")
        stream = mock.AsyncMock(return_value="不应调用")
        self.controller.backend = types.SimpleNamespace(
            _rule_prompt="",
            get_response_mode=lambda: "complete",
            _request_chat_completion=complete,
            request_streaming_chat_completion=stream,
        )
        self.controller._wait_for_xiaoai_asr_text = mock.AsyncMock(
            return_value="问题"
        )
        self.controller._play_tts = mock.AsyncMock()
        self.controller._play_notify = mock.AsyncMock()

        with mock.patch.object(
            self.streaming_module,
            "SequentialSpeechQueue",
            side_effect=AssertionError("complete must not create queue"),
        ):
            result = asyncio.run(
                self.controller._run_one_turn_with_xiaoai_asr()
            )

        self.assertEqual("continue", result)
        complete.assert_awaited_once_with("问题")
        stream.assert_not_awaited()
        self.controller._play_tts.assert_awaited_once_with("完整回答")

    def test_complete_failure_is_not_retried_as_streaming(self):
        complete = mock.AsyncMock(
            side_effect=RuntimeError("request failed")
        )
        stream = mock.AsyncMock(return_value="不应调用")
        self.controller.backend = types.SimpleNamespace(
            _rule_prompt="",
            get_response_mode=lambda: "complete",
            _request_chat_completion=complete,
            request_streaming_chat_completion=stream,
        )

        with self.assertRaisesRegex(RuntimeError, "request failed"):
            asyncio.run(
                self.controller._request_backend_turn(
                    "问题",
                    play_send_sound=False,
                )
            )

        complete.assert_awaited_once_with("问题")
        stream.assert_not_awaited()

    def test_complete_http_wait_is_cancelled_by_stop(self):
        async def scenario():
            request_started = asyncio.Event()
            request_cancelled = asyncio.Event()

            async def complete(_text):
                request_started.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    request_cancelled.set()
                    raise

            self.controller.config.values["hermes"]["input_mode"] = (
                "local_asr"
            )
            self.controller.active = True
            self.controller._loop = asyncio.get_running_loop()
            self.controller.backend = types.SimpleNamespace(
                _rule_prompt="",
                get_response_mode=lambda: "complete",
                _request_chat_completion=complete,
            )
            turn = asyncio.create_task(
                self.controller._request_backend_turn(
                    "问题",
                    play_send_sound=False,
                )
            )
            await request_started.wait()
            self.controller.stop()
            with self.assertRaises(asyncio.CancelledError):
                await turn
            return request_cancelled.is_set()

        self.assertTrue(asyncio.run(scenario()))

    def test_complete_tts_playback_token_is_stopped_without_replay(self):
        async def scenario():
            playback_started = asyncio.Event()
            release_playback = asyncio.Event()
            calls = []

            async def play_response(text, **_kwargs):
                calls.append(text)
                playback_started.set()
                await release_playback.wait()

            self.controller.config.values["hermes"]["input_mode"] = (
                "local_asr"
            )
            self.controller.active = True
            self.controller._loop = asyncio.get_running_loop()
            self.controller.backend = types.SimpleNamespace(
                _play_response_with_tts=play_response,
                get_tts_speaker_for_session_key=lambda: "xiaoai",
            )
            external_module = importlib.import_module(
                "core.external_conversation"
            )
            with (
                mock.patch.object(
                    self.module.open_xiaoai_server,
                    "begin_playback_session",
                    return_value=73,
                ),
                mock.patch.object(
                    external_module.open_xiaoai_server,
                    "stop_tts_playback",
                    create=True,
                ) as stop_playback,
            ):
                playback = asyncio.create_task(
                    self.controller._play_tts("完整回答")
                )
                await playback_started.wait()
                self.controller.stop()
                stop_playback.assert_called_once_with(73)
                release_playback.set()
                await playback
            return calls

        self.assertEqual(["完整回答"], asyncio.run(scenario()))

    def test_router_none_is_not_treated_as_playback_failure(self):
        backend = types.SimpleNamespace(
            _play_response_with_tts=mock.AsyncMock(
                return_value=None
            ),
            get_tts_speaker_for_session_key=mock.Mock(
                return_value="xiaoai"
            ),
        )
        self.controller.backend = backend

        self.assertIsNone(asyncio.run(self.controller._play_tts("等待调用结束")))
        backend._play_response_with_tts.assert_awaited_once()
        self.assertIsNone(self.controller._playback_token)

    def test_unhandled_tts_error_propagates_without_retry(self):
        backend = types.SimpleNamespace(
            _play_response_with_tts=mock.AsyncMock(
                side_effect=RuntimeError("unhandled TTS error")
            ),
            get_tts_speaker_for_session_key=mock.Mock(return_value="xiaoai"),
        )
        self.controller.backend = backend

        async def scenario():
            with self.assertRaisesRegex(
                RuntimeError,
                "unhandled TTS error",
            ):
                await self.controller._play_tts("播放失败")

        asyncio.run(scenario())
        backend._play_response_with_tts.assert_awaited_once()
        self.assertIsNone(self.controller._playback_token)


async def _append_async(items, value):
    items.append(value)


if __name__ == "__main__":
    unittest.main()
