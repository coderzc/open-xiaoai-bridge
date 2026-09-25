import asyncio
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch


sys.modules.setdefault("open_xiaoai_server", types.SimpleNamespace())

router_module = importlib.import_module("core.services.tts.router")
TTSRouter = router_module.TTSRouter


class TTSRouterProviderTest(unittest.TestCase):
    def test_explicit_provider_is_shared_by_all_backends(self):
        for provider in ("xiaoai", "doubao", "openai", "mlx_audio"):
            self.assertEqual(
                provider,
                TTSRouter.resolve_provider(provider, "xiaoai"),
            )

    def test_missing_provider_preserves_legacy_speaker_selection(self):
        self.assertEqual("xiaoai", TTSRouter.resolve_provider(None, "xiaoai"))
        self.assertEqual("doubao", TTSRouter.resolve_provider(None, None))
        self.assertEqual("doubao", TTSRouter.resolve_provider(None, "alloy"))


class BackendTTSProviderConfigTest(unittest.TestCase):
    def test_openclaw_reads_tts_provider_without_changing_existing_speaker_keys(self):
        module = importlib.import_module("core.openclaw")
        manager = module.OpenClawManager
        config = {
            "tts_provider": "mlx_audio",
            "tts_speaker": "xiaoai",
            "agent_tts_speakers": {"assistant": "xiaoai"},
        }
        config_manager = types.SimpleNamespace(
            get_app_config=lambda *_args: config,
            add_reload_listener=lambda *_args: None,
        )
        previous_listener_state = manager._reload_listener_registered
        try:
            manager._reload_listener_registered = False
            with patch.object(
                module.ConfigManager, "instance", return_value=config_manager
            ):
                manager.reload_from_config(enabled=True)
            self.assertEqual("mlx_audio", manager._tts_provider)
            self.assertEqual("xiaoai", manager._tts_speaker)
            self.assertEqual({"assistant": "xiaoai"}, manager._agent_tts_speakers)
        finally:
            manager._reload_listener_registered = previous_listener_state

    def test_qwenpaw_reads_tts_provider_without_changing_existing_speaker_keys(self):
        module = importlib.import_module("core.qwenpaw")
        manager = module.QwenPawManager
        config = {
            "tts_provider": "openai",
            "tts_speaker": "sage",
            "session_tts_speakers": {"agent:default:main": "sage"},
        }
        config_manager = types.SimpleNamespace(
            get_app_config=lambda *_args: config,
            add_reload_listener=lambda *_args: None,
        )
        previous_listener_state = manager._reload_listener_registered
        try:
            manager._reload_listener_registered = False
            with patch.object(
                module.ConfigManager, "instance", return_value=config_manager
            ):
                manager.reload_from_config(enabled=True)
            self.assertEqual("openai", manager._tts_provider)
            self.assertEqual("sage", manager._tts_speaker)
            self.assertEqual(
                {"agent:default:main": "sage"}, manager._session_tts_speakers
            )
        finally:
            manager._reload_listener_registered = previous_listener_state


class BackendTTSRoutingTest(unittest.IsolatedAsyncioTestCase):
    async def test_openclaw_delegates_to_shared_router(self):
        module = importlib.import_module("core.openclaw")
        manager = module.OpenClawManager
        previous = (
            manager._tts_provider,
            manager._tts_speaker,
            manager._tts_speed,
            manager._session_key,
            manager._initialized,
        )
        manager._tts_provider = "mlx_audio"
        manager._tts_speaker = "xiaoai"
        manager._tts_speed = 1.1
        manager._session_key = "agent:assistant:main"
        manager._initialized = True
        try:
            with patch.object(
                router_module.TTSRouter, "play", new=AsyncMock()
            ) as play:
                await manager._play_response_with_tts("你好", playback_token=7)
            play.assert_awaited_once_with(
                "你好",
                configured_provider="mlx_audio",
                tts_speaker="xiaoai",
                tts_speed=1.1,
                playback_token=7,
                log_prefix="OpenClaw",
            )
        finally:
            (
                manager._tts_provider,
                manager._tts_speaker,
                manager._tts_speed,
                manager._session_key,
                manager._initialized,
            ) = previous

    async def test_qwenpaw_delegates_to_shared_router(self):
        module = importlib.import_module("core.qwenpaw")
        manager = module.QwenPawManager
        previous = (
            manager._tts_provider,
            manager._tts_speaker,
            manager._tts_speed,
        )
        manager._tts_provider = "openai"
        manager._tts_speaker = "sage"
        manager._tts_speed = 0.9
        try:
            with patch.object(
                router_module.TTSRouter, "play", new=AsyncMock()
            ) as play:
                await manager._play_response_with_tts("你好", playback_token=8)
            play.assert_awaited_once_with(
                "你好",
                configured_provider="openai",
                tts_speaker="sage",
                tts_speed=0.9,
                playback_token=8,
                log_prefix="QwenPaw",
            )
        finally:
            manager._tts_provider, manager._tts_speaker, manager._tts_speed = previous


class TTSRouterUpstreamContractTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.speaker = types.SimpleNamespace(
            play=AsyncMock(return_value=True),
            play_server_file=AsyncMock(return_value=True),
        )
        self.enterContext(patch.object(
            importlib.import_module("core.ref"), "get_speaker", return_value=self.speaker,
        ))
        self.config = self.enterContext(patch.object(
            router_module.ConfigManager.instance(), "get_app_config",
            return_value={"base_url": "http://tts.test/v1", "response_format": "wav"},
        ))
        self.synthesize = self.enterContext(patch.object(
            router_module.OpenAITTS, "synthesize", new=AsyncMock(return_value=b"audio"),
        ))

    async def play(self, provider="openai", token=71):
        return await TTSRouter.play(
            "一句话", configured_provider=provider, tts_speaker="xiaoai",
            playback_token=token,
        )

    async def test_native_discards_boolean_and_retries_once_on_exception(self):
        for value in (True, False):
            with self.subTest(value=value):
                self.speaker.play.reset_mock()
                self.speaker.play.return_value = value
                self.assertIsNone(await self.play("xiaoai"))
                self.speaker.play.assert_awaited_once()
        self.speaker.play.reset_mock()
        self.speaker.play.side_effect = [RuntimeError("native failed"), True]
        self.assertIsNone(await self.play("xiaoai"))
        self.assertEqual(2, self.speaker.play.await_count)

    async def test_missing_speaker_does_not_provide_a_failure_result(self):
        with patch.object(importlib.import_module("core.ref"), "get_speaker", return_value=None):
            self.assertIsNone(await self.play("xiaoai"))
            self.assertIsNone(await self.play())

    async def test_doubao_none_means_normal_interface_completion(self):
        with patch.object(TTSRouter, "_play_doubao", new=AsyncMock(return_value=None)) as play:
            self.assertIsNone(await self.play("doubao"))
            self.assertEqual(71, play.await_args.kwargs["playback_token"])
        self.speaker.play.assert_not_awaited()

    async def test_file_providers_keep_upstream_arguments_and_clean_file(self):
        for provider in ("openai", "mlx_audio"):
            with self.subTest(provider=provider):
                async def play_file(*, file_path, blocking):
                    self.assertTrue(Path(file_path).is_file())
                    self.assertTrue(blocking)
                    return True
                self.speaker.play_server_file.side_effect = play_file
                self.assertIsNone(await self.play(provider))
                path = self.speaker.play_server_file.await_args.kwargs["file_path"]
                self.assertFalse(Path(path).exists())
        self.speaker.play.assert_not_awaited()

    async def test_provider_failure_falls_back_once_without_returning_result(self):
        for fallback_result in (True, False, RuntimeError("fallback failed")):
            with self.subTest(fallback_result=fallback_result):
                self.speaker.play.reset_mock()
                self.speaker.play.side_effect = (
                    fallback_result if isinstance(fallback_result, Exception) else None
                )
                self.speaker.play.return_value = fallback_result
                self.synthesize.side_effect = TimeoutError("synthesis failed")
                self.assertIsNone(await self.play())
                self.speaker.play.assert_awaited_once_with(text="一句话", blocking=True)

    async def test_file_failure_cleans_before_single_fallback(self):
        for failure in (False, RuntimeError("file failed")):
            with self.subTest(failure=failure):
                self.speaker.play.reset_mock()
                self.speaker.play_server_file.side_effect = (
                    failure if isinstance(failure, Exception) else None
                )
                self.speaker.play_server_file.return_value = failure
                async def fallback(**_kwargs):
                    path = self.speaker.play_server_file.await_args.kwargs["file_path"]
                    self.assertFalse(Path(path).exists())
                    return True
                self.speaker.play.side_effect = fallback
                self.assertIsNone(await self.play())
                self.speaker.play.assert_awaited_once()

    async def test_cancellation_during_synthesis_file_or_fallback_propagates(self):
        for stage in ("synthesis", "file", "fallback"):
            with self.subTest(stage=stage):
                self.speaker.play.reset_mock()
                self.speaker.play_server_file.reset_mock()
                self.speaker.play.side_effect = None
                self.speaker.play_server_file.side_effect = None
                self.synthesize.side_effect = None
                target = self.synthesize if stage == "synthesis" else self.speaker.play_server_file
                if stage == "fallback":
                    self.synthesize.side_effect = RuntimeError("provider failed")
                    target = self.speaker.play
                started = asyncio.Event()
                async def block(*_args, **_kwargs):
                    started.set()
                    await asyncio.Future()
                target.side_effect = block
                task = asyncio.create_task(self.play())
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertEqual(int(stage == "fallback"), self.speaker.play.await_count)
                if stage == "file":
                    path = self.speaker.play_server_file.await_args.kwargs["file_path"]
                    self.assertFalse(Path(path).exists())

    async def test_native_and_file_do_not_require_a_native_token_query(self):
        # 上游扩展没有公开 token 查询；不能依赖 fork 新增的接口。
        with patch.object(router_module, "open_xiaoai_server", types.SimpleNamespace()):
            for token in (None, 71):
                self.assertIsNone(await self.play("xiaoai", token=token))
                self.assertIsNone(await self.play(token=token))


class ExistingBackendTTSConversationTest(unittest.IsolatedAsyncioTestCase):
    """使用真实旧入口、controller 和 Router，只替换模型与设备 I/O。"""

    async def _run_case(self, name, prefix, scenario, input_mode):
        manager = getattr(importlib.import_module(f"core.{name}"), f"{prefix}Manager")
        controller_class = getattr(
            importlib.import_module(f"core.{name}_conversation"),
            f"{prefix}ConversationController",
        )
        external = importlib.import_module("core.external_conversation")
        current_token = 0

        class Backend(manager):
            _initialized = True
            _tts_provider = "xiaoai"
            _tts_speaker = "xiaoai"
            _tts_speed = 1.0
            _session_key = "test-speaker"
            _session_tts_speakers = {}
            _agent_tts_speakers = {}

        def begin():
            nonlocal current_token
            current_token += 1
            return current_token

        native = types.SimpleNamespace(
            begin_playback_session=Mock(side_effect=begin),
            is_playback_session_active=Mock(
                side_effect=lambda token: token == current_token,
            ),
        )
        attempts = 0

        async def play(**_kwargs):
            nonlocal attempts
            attempts += 1
            if scenario == "exception" and attempts == 1:
                raise RuntimeError("native failed")
            if scenario == "replacement":
                # 模拟独立播放替换 token，不冒充一次用户唤醒或 task.cancel。
                native.begin_playback_session()
            return True

        speaker = types.SimpleNamespace(play=AsyncMock(side_effect=play))
        controller = controller_class.__new__(controller_class)
        controller.backend = Backend
        controller.config = types.SimpleNamespace(get_app_config=lambda _key, default=None: default)
        controller._playback_token = None
        controller._wait_for_xiaoai_asr_text = AsyncMock(return_value="问题")
        controller._request_backend_turn = AsyncMock(return_value=("回答", False))
        controller._wait_for_speech = AsyncMock(return_value=b"audio")
        controller._wait_for_silence = AsyncMock()
        controller._stop_recording = AsyncMock()
        controller._start_recording = AsyncMock()
        controller._play_notify = AsyncMock()
        asr = types.SimpleNamespace(
            ASRService=types.SimpleNamespace(asr=lambda *_args, **_kwargs: "问题"),
        )

        with (
            patch.object(external, "open_xiaoai_server", native),
            patch.object(router_module, "open_xiaoai_server", native),
            patch.object(importlib.import_module("core.ref"), "get_speaker", return_value=speaker),
            patch.object(external, "get_speaker", return_value=speaker),
            patch.object(external, "get_vad", return_value=object()),
            patch.dict(sys.modules, {"core.services.audio.asr": asr}),
        ):
            run = getattr(controller, f"_run_one_turn_with_{input_mode}")
            self.assertEqual("continue", await run())

        self.assertEqual(2 if scenario == "exception" else 1, speaker.play.await_count)
        controller._stop_recording.assert_awaited_once()
        controller._play_notify.assert_awaited_once()
        controller._start_recording.assert_awaited_once()
        controller._request_backend_turn.assert_awaited_once()
        self.assertIsNone(controller._playback_token)
        native.is_playback_session_active.assert_not_called()

    async def test_native_exception_keeps_upstream_retry_and_listening(self):
        for name, prefix in (("openai", "OpenAI"), ("openclaw", "OpenClaw"), ("qwenpaw", "QwenPaw")):
            for input_mode in ("local_asr", "xiaoai_asr"):
                with self.subTest(backend=name, input_mode=input_mode):
                    await self._run_case(name, prefix, "exception", input_mode)

    async def test_replacement_token_does_not_cancel_old_backend_listening(self):
        for name, prefix in (("openai", "OpenAI"), ("openclaw", "OpenClaw"), ("qwenpaw", "QwenPaw")):
            for input_mode in ("local_asr", "xiaoai_asr"):
                with self.subTest(backend=name, input_mode=input_mode):
                    await self._run_case(name, prefix, "replacement", input_mode)


if __name__ == "__main__":
    unittest.main()
