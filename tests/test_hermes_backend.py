import ast
import asyncio
import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class _Config:
    def __init__(self, values):
        self.values = values
        self.listeners = []

    def get_app_config(self, path, default=None):
        value = self.values
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
        return value

    def add_reload_listener(self, listener):
        self.listeners.append(listener)


class HermesBackendTest(unittest.TestCase):
    def setUp(self):
        sys.modules["open_xiaoai_server"] = types.SimpleNamespace()
        if isinstance(sys.modules.get("aiohttp"), types.SimpleNamespace):
            sys.modules.pop("aiohttp", None)
        importlib.import_module("aiohttp")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        for name in ("core.openai", "core.hermes"):
            sys.modules.pop(name, None)
        self.openai_module = importlib.import_module("core.openai")
        self.openai = self.openai_module.OpenAIManager
        self.hermes_module = importlib.import_module("core.hermes")
        self.hermes = self.hermes_module.HermesManager

    def tearDown(self):
        self.openai._sessions.clear()
        self.hermes._sessions.clear()
        self.hermes._hermes_session_ids.clear()
        self.hermes._profile = ""
        self.hermes._profile_api_keys = {}
        self.hermes._response_mode = self.hermes.DEFAULT_RESPONSE_MODE
        self.hermes._capabilities_checked = False
        self.hermes._capabilities = None
        self.hermes._capabilities_diagnostic = None

    def test_openai_and_hermes_runtime_state_are_isolated(self):
        self.openai._session_key = "plain-openai"
        self.openai._session_header = "X-Hermes-Session-Key"
        self.openai._sessions["plain-openai"] = [
            {"role": "user", "content": "OpenAI history"}
        ]
        self.hermes._session_key = "agent:hermes:speaker"
        self.hermes._session_header = "X-Hermes-Session-Key"

        self.assertEqual(
            "plain-openai",
            self.openai._headers()["X-Hermes-Session-Key"],
        )
        self.assertEqual(
            "agent:hermes:speaker",
            self.hermes._headers()["X-Hermes-Session-Key"],
        )
        self.assertEqual({}, self.hermes._sessions)

    def test_hermes_native_session_id_is_reused_within_conversation(self):
        self.hermes._session_key = "agent:hermes:speaker"
        scope_key = self.hermes._conversation_scope_key()

        self.hermes._capture_response_headers(
            {"X-Hermes-Session-Id": "session-123"},
            session_key=scope_key,
        )

        self.assertEqual(
            "session-123",
            self.hermes._headers()["X-Hermes-Session-Id"],
        )

    def test_explicit_reset_discards_text_and_native_session(self):
        self.hermes._session_key = "agent:hermes:speaker"
        scope_key = self.hermes._conversation_scope_key()
        self.hermes._sessions[scope_key] = [
            {"role": "assistant", "content": "昨天已经播放"}
        ]
        self.hermes._hermes_session_ids[scope_key] = (
            "session-yesterday"
        )

        self.hermes.reset_session()

        self.assertNotIn(
            scope_key,
            self.hermes._sessions,
        )
        self.assertNotIn(
            scope_key,
            self.hermes._hermes_session_ids,
        )

    def test_hermes_does_not_add_per_wake_session_rotation(self):
        source = (ROOT / "core/hermes.py").read_text(encoding="utf-8")
        controller_source = (
            ROOT / "core/hermes_conversation.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("begin_conversation", source)
        self.assertNotIn("begin_conversation", controller_source)
        self.assertNotIn("bridge-voice-", source)

    def test_complete_response_uses_stream_false_and_captures_session(self):
        posts = []

        class FakeResponse:
            status = 200
            headers = {"X-Hermes-Session-Id": "session-complete"}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def json(self, **_kwargs):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "完整回答"
                            }
                        }
                    ]
                }

        class FakeSession:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def post(self, *_args, **kwargs):
                posts.append(kwargs)
                return FakeResponse()

        self.hermes._session_key = "agent:hermes:complete"
        self.hermes._model = "hermes-agent"
        self.hermes._extra_body = {}
        self.hermes._temperature = None
        self.hermes._max_tokens = None
        self.hermes._timeout = 10

        async def scenario():
            with (
                mock.patch.object(
                    self.hermes_module.aiohttp,
                    "ClientSession",
                    FakeSession,
                ),
                mock.patch.object(
                    self.hermes_module.aiohttp,
                    "ClientTimeout",
                    lambda **_kwargs: object(),
                ),
            ):
                return await self.hermes._request_chat_completion(
                    "问题"
                )

        self.assertEqual("完整回答", asyncio.run(scenario()))
        self.assertEqual(1, len(posts))
        self.assertIs(False, posts[0]["json"]["stream"])
        self.assertEqual(
            "session-complete",
            self.hermes._hermes_session_ids[
                self.hermes._conversation_scope_key()
            ],
        )
        self.assertEqual(
            [
                {"role": "user", "content": "问题"},
                {"role": "assistant", "content": "完整回答"},
            ],
            self.hermes._sessions[
                self.hermes._conversation_scope_key()
            ],
        )

    def test_profiles_use_official_url_prefix_and_isolate_sessions(self):
        self.hermes._base_url = "http://hermes.test/v1"
        self.hermes._api_key = "default-key"
        self.hermes._profile_api_keys = {"coder": "coder-key"}
        self.hermes._session_key = "speaker"

        default_scope = self.hermes._conversation_scope_key()
        self.hermes._sessions[default_scope] = [{"role": "user", "content": "a"}]
        self.hermes.set_profile("coder")
        coder_scope = self.hermes._conversation_scope_key()

        self.assertEqual(
            "http://hermes.test/p/coder/v1/chat/completions",
            self.hermes._chat_completions_url(),
        )
        self.assertEqual(
            "Bearer coder-key",
            self.hermes._headers()["Authorization"],
        )
        self.assertNotEqual(default_scope, coder_scope)
        self.assertEqual([], self.hermes._sessions.get(coder_scope, []))

    def test_named_profile_requires_its_own_api_key(self):
        with self.assertRaisesRegex(ValueError, "No API key configured"):
            self.hermes.set_profile("coder")

    def test_main_app_exposes_session_controls(self):
        from core import app as app_module

        MainApp = app_module.MainApp
        manager = app_module.HermesManager

        with (
            mock.patch.object(manager, "set_session_key") as set_key,
            mock.patch.object(manager, "reset_session") as reset,
            mock.patch.object(
                manager,
                "get_session_state",
                return_value={"session_key": "speaker"},
            ),
            mock.patch.object(manager, "set_profile") as set_profile,
        ):
            MainApp.set_hermes_session_key(object(), "speaker")
            MainApp.reset_hermes_session(object(), "speaker")
            state = MainApp.get_hermes_session_state(object())
            MainApp.set_hermes_profile(object(), "coder")

        set_key.assert_called_once_with("speaker")
        reset.assert_called_once_with("speaker")
        set_profile.assert_called_once_with("coder")
        self.assertEqual({"session_key": "speaker"}, state)

    def test_agent_autonomous_path_does_not_auto_play_final(self):
        from core import app as app_module

        MainApp = app_module.MainApp
        manager = app_module.HermesManager
        manager._response_mode = "complete"
        manager._rule_prompt_for_skill = "Use xiaoai-tts when needed."
        with (
            mock.patch.object(
                manager,
                "send",
                new=mock.AsyncMock(return_value="run-1"),
            ) as send,
            mock.patch.object(
                manager,
                "send_and_play_reply",
                new=mock.AsyncMock(),
            ) as send_and_play,
        ):
            result = asyncio.run(
                MainApp.send_to_hermes(object(), "提醒我喝水")
            )

        self.assertEqual("run-1", result)
        send.assert_awaited_once_with(
            "提醒我喝水\nUse xiaoai-tts when needed.",
            wait_response=False,
        )
        send_and_play.assert_not_awaited()

    def test_hermes_reads_only_its_own_configuration(self):
        config = _Config(
            {
                "openai": {
                    "base_url": "http://openai.test/v1",
                    "model": "plain-model",
                },
                "hermes": {
                    "base_url": "http://hermes.test/v1",
                    "model": "hermes-model",
                    "session_key": "agent:hermes:test",
                },
            }
        )
        self.hermes._reload_listener_registered = False

        with (
            mock.patch.object(
                self.hermes_module.ConfigManager,
                "instance",
                return_value=config,
            ),
            mock.patch.object(
                self.hermes_module,
                "get_env",
                return_value="true",
            ),
        ):
            self.hermes.reload_from_config()

        self.assertTrue(self.hermes._enabled)
        self.assertEqual("http://hermes.test/v1", self.hermes._base_url)
        self.assertEqual("hermes-model", self.hermes._model)
        self.assertEqual("agent:hermes:test", self.hermes._session_key)
        self.assertEqual("X-Hermes-Session-Key", self.hermes._session_header)
        self.assertEqual("streaming", self.hermes.get_response_mode())

    def test_response_mode_is_normalized_and_invalid_value_warns(self):
        config = _Config(
            {"hermes": {"response_mode": "  COMPLETE  "}}
        )
        self.hermes._reload_listener_registered = False
        with (
            mock.patch.object(
                self.hermes_module.ConfigManager,
                "instance",
                return_value=config,
            ),
            mock.patch.object(
                self.hermes_module,
                "get_env",
                return_value="true",
            ),
        ):
            self.hermes.reload_from_config()
        self.assertEqual("complete", self.hermes.get_response_mode())

        config.values["hermes"]["response_mode"] = "automatic"
        with (
            mock.patch.object(
                self.hermes_module.ConfigManager,
                "instance",
                return_value=config,
            ),
            mock.patch.object(
                self.hermes_module,
                "get_env",
                return_value="true",
            ),
            mock.patch.object(
                self.hermes_module.logger,
                "warning",
            ) as warning,
        ):
            self.hermes.reload_from_config()
        self.assertEqual("streaming", self.hermes.get_response_mode())
        warning.assert_called_once()

    def test_tts_provider_is_independent_and_removal_restores_legacy_rule(self):
        config = _Config({"hermes": {}, "openai": {"tts_provider": "openai"}})
        self.openai._tts_provider = "openai"
        self.assertIn("_tts_provider", self.hermes.__dict__)
        self.hermes._reload_listener_registered = True
        with mock.patch.object(self.hermes_module.ConfigManager, "instance", return_value=config):
            for provider in ("xiaoai", "doubao", "openai", "mlx_audio"):
                config.values["hermes"]["tts_provider"] = " " + provider.upper() + " "
                self.hermes.reload_from_config(enabled=True)
                self.assertEqual(provider, self.hermes._tts_provider)
                self.assertEqual("openai", self.openai._tts_provider)
            del config.values["hermes"]["tts_provider"]
            self.hermes.reload_from_config(enabled=True)
        self.assertIsNone(self.hermes._tts_provider)
        self.assertEqual("xiaoai", self.hermes._resolve_tts_provider("xiaoai"))
        self.assertEqual("doubao", self.hermes._resolve_tts_provider("voice"))

    def test_hermes_delegates_voice_token_and_void_contract_to_router(self):
        from core.services.tts.router import TTSRouter

        self.hermes._tts_speaker = "default-voice"
        self.hermes._session_key = "speaker"
        self.hermes._session_tts_speakers = {"speaker": "session-voice"}
        self.hermes._tts_speed = 1.1
        for provider in ("xiaoai", "doubao", "openai", "mlx_audio"):
            self.hermes._tts_provider = provider
            for override, voice in ((None, "session-voice"), ("explicit", "explicit")):
                with mock.patch.object(TTSRouter, "play", new=mock.AsyncMock(return_value=None)) as play:
                    actual = asyncio.run(self.hermes._play_response_with_tts(
                        "回答", tts_speaker=override, playback_token=17,
                    ))
                self.assertIsNone(actual)
                play.assert_awaited_once_with(
                    "回答", configured_provider=provider, tts_speaker=voice,
                    tts_speed=1.1, playback_token=17, log_prefix="Hermes",
                )

    def test_response_mode_does_not_change_session_scope_or_native_id(self):
        self.hermes._session_key = "speaker"
        self.hermes._profile = ""
        scope_key = self.hermes._conversation_scope_key()
        self.hermes._sessions[scope_key] = [
            {"role": "assistant", "content": "已有上下文"}
        ]
        self.hermes._hermes_session_ids[scope_key] = "native-session"

        self.hermes._response_mode = "streaming"
        streaming_scope = self.hermes._conversation_scope_key()
        self.hermes._response_mode = "complete"
        complete_scope = self.hermes._conversation_scope_key()

        self.assertEqual(scope_key, streaming_scope)
        self.assertEqual(scope_key, complete_scope)
        self.assertEqual(
            "native-session",
            self.hermes._headers()["X-Hermes-Session-Id"],
        )
        self.assertEqual(1, len(self.hermes._sessions[scope_key]))

    def test_main_app_adds_hermes_after_existing_backend_arguments(self):
        tree = ast.parse((ROOT / "core/app.py").read_text(encoding="utf-8"))
        main_app = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MainApp"
        )
        expected = [
            "enable_xiaozhi",
            "enable_openclaw",
            "enable_openai",
            "enable_qwenpaw",
            "enable_hermes",
        ]
        for method_name in ("instance", "__init__"):
            method = next(
                node
                for node in main_app.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == method_name
            )
            argument_names = [argument.arg for argument in method.args.args]
            self.assertEqual(expected, argument_names[-len(expected):])

    def test_generic_openai_source_remains_upstream_compatible(self):
        source = (ROOT / "core/openai.py").read_text(encoding="utf-8")
        self.assertNotIn("hermes.tool.progress", source)
        self.assertNotIn("HERMES_ENABLE", source)
        self.assertNotIn("HermesManager", source)

    def test_only_hermes_interprets_hermes_progress_events(self):
        progress = []

        async def fake_stream(
            _manager,
            _text,
            *,
            on_delta,
            on_event,
            log_name,
        ):
            del on_delta, log_name
            event = types.SimpleNamespace(
                event="hermes.tool.progress",
                data=(
                    '{"tool":"weather","status":"running",'
                    '"toolCallId":"call-1"}'
                ),
            )
            await on_event(event)
            return "完成"

        async def scenario():
            with mock.patch.object(
                self.hermes_module,
                "stream_openai_chat_completion",
                side_effect=fake_stream,
            ):
                return await self.hermes.request_streaming_chat_completion(
                    "天气怎么样",
                    on_delta=lambda _value: None,
                    on_tool_progress=lambda event: progress.append(event),
                )

        self.assertEqual("完成", asyncio.run(scenario()))
        self.assertEqual(["weather"], [event.tool for event in progress])

    def test_default_routes_send_hermes_commands_directly_to_hermes(self):
        spec = importlib.util.spec_from_file_location(
            "hermes_route_config_test",
            ROOT / "config.py",
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        speaker = types.SimpleNamespace(
            play=mock.AsyncMock(),
            abort_xiaoai=mock.AsyncMock(),
        )
        app = types.SimpleNamespace()

        kws_result = asyncio.run(
            config_module.before_wakeup(
                speaker,
                "你好赫尔墨斯",
                "kws",
                app,
            )
        )
        xiaoai_result = asyncio.run(
            config_module.before_wakeup(
                speaker,
                "召唤赫尔墨斯",
                "xiaoai",
                app,
            )
        )

        self.assertEqual("hermes", kws_result)
        self.assertEqual("hermes", xiaoai_result)
        speaker.abort_xiaoai.assert_awaited_once()

    def test_generic_openai_defaults_do_not_gain_streaming_options(self):
        spec = importlib.util.spec_from_file_location(
            "hermes_config_boundary_test",
            ROOT / "config.py",
        )
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        openai_config = config_module.APP_CONFIG["openai"]
        hermes_config = config_module.APP_CONFIG["hermes"]

        self.assertEqual(
            "X-Hermes-Session-Key",
            openai_config["session_header"],
        )
        self.assertNotIn("streaming", openai_config)
        self.assertNotIn("progress", openai_config)
        self.assertEqual("streaming", hermes_config["response_mode"])
        self.assertNotIn("enabled", hermes_config["streaming"])
        self.assertTrue(hermes_config["progress"]["enabled"])


if __name__ == "__main__":
    unittest.main()
