import asyncio
import importlib
import importlib.util
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class WakeupKeywordStartupTest(unittest.TestCase):
    def test_keyword_generation_enabled_for_openclaw(self):
        spec = importlib.util.spec_from_file_location(
            "kws_keywords_for_test",
            ROOT / "core/services/audio/kws/keywords.py",
        )
        keywords = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(keywords)

        class ConfigManagerStub:
            @classmethod
            def instance(cls):
                return cls()

            def get_app_config(self, path=None, default=None):
                if path == "homeassistant.enabled":
                    return False
                return default

        with (
            mock.patch.object(keywords, "ConfigManager", ConfigManagerStub),
            mock.patch.dict(
                os.environ,
                {
                    "OPENCLAW_ENABLE": "1",
                    "OPENCLAW_ENABLED": "",
                },
                clear=False,
            ),
        ):
            should_run, reason = keywords.should_generate_keywords()

        self.assertTrue(should_run)
        self.assertEqual(reason, "")

    def test_keyword_generation_enabled_for_homeassistant(self):
        """Home Assistant has no env var flag — should_generate_keywords()
        must read homeassistant.enabled from config.py directly, or a
        'Home Assistant only' setup would never get a keywords.txt."""
        spec = importlib.util.spec_from_file_location(
            "kws_keywords_for_test",
            ROOT / "core/services/audio/kws/keywords.py",
        )
        keywords = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(keywords)

        class ConfigManagerStub:
            @classmethod
            def instance(cls):
                return cls()

            def get_app_config(self, path=None, default=None):
                if path == "homeassistant.enabled":
                    return True
                return default

        with (
            mock.patch.object(keywords, "ConfigManager", ConfigManagerStub),
            mock.patch.dict(
                os.environ,
                {
                    "OPENCLAW_ENABLE": "",
                    "OPENCLAW_ENABLED": "",
                },
                clear=False,
            ),
        ):
            should_run, reason = keywords.should_generate_keywords()

        self.assertTrue(should_run)
        self.assertEqual(reason, "")

    def test_keyword_generation_disabled_when_both_off(self):
        spec = importlib.util.spec_from_file_location(
            "kws_keywords_for_test",
            ROOT / "core/services/audio/kws/keywords.py",
        )
        keywords = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(keywords)

        class ConfigManagerStub:
            @classmethod
            def instance(cls):
                return cls()

            def get_app_config(self, path=None, default=None):
                if path == "homeassistant.enabled":
                    return False
                return default

        with (
            mock.patch.object(keywords, "ConfigManager", ConfigManagerStub),
            mock.patch.dict(
                os.environ,
                {
                    "OPENCLAW_ENABLE": "",
                    "OPENCLAW_ENABLED": "",
                },
                clear=False,
            ),
        ):
            should_run, _reason = keywords.should_generate_keywords()

        self.assertFalse(should_run)

    def test_startup_entrypoints_prepare_keywords_for_openclaw_and_homeassistant(self):
        start_sh = (ROOT / "scripts/start.sh").read_text(encoding="utf8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf8")

        self.assertIn("OPENCLAW_ENABLE_VALUE", start_sh)
        self.assertIn("HOMEASSISTANT_ENABLED", start_sh)
        self.assertIn('[[ "$OPENCLAW_ENABLE_VALUE" =~ ^(1|true|yes)$ ]]', start_sh)
        self.assertIn('[[ "$HOMEASSISTANT_ENABLED" == "true" ]]', start_sh)
        self.assertIn("homeassistant", start_sh)
        # No leftover XiaoZhi/OpenAI/QwenPaw gating in either entrypoint.
        self.assertNotIn("XIAOZHI_ENABLE", start_sh)
        self.assertNotIn("OPENAI_ENABLE", start_sh)
        self.assertNotIn("QWENPAW_ENABLE", start_sh)
        self.assertNotIn("XIAOZHI_ENABLE", dockerfile)
        self.assertNotIn("OPENAI_ENABLE", dockerfile)
        self.assertNotIn("QWENPAW_ENABLE", dockerfile)
        self.assertIn('python core/services/audio/kws/keywords.py', dockerfile)


class XiaoAIWakeupKeywordTest(unittest.TestCase):
    def test_custom_xiaoai_asr_wakeup_dispatches_to_keyword_flow(self):
        np_stub = types.SimpleNamespace(int16=object(), float32=object())
        server_stub = types.SimpleNamespace()

        class ConfigManagerStub:
            @classmethod
            def instance(cls):
                return cls()

            def get_app_config(self, path=None, default=None):
                if path == "wakeup.keywords":
                    return ["你好小黑"]
                if path == "xiaoai":
                    return {}
                return default

        config_stub = types.SimpleNamespace(ConfigManager=ConfigManagerStub)

        isolated_modules = ("core.xiaoai", "core.wakeup_session")
        saved_modules = {
            module_name: sys.modules.get(module_name)
            for module_name in isolated_modules
        }
        for module_name in isolated_modules:
            sys.modules.pop(module_name, None)

        try:
            with mock.patch.dict(
                sys.modules,
                {
                    "numpy": np_stub,
                    "open_xiaoai_server": server_stub,
                    "core.utils.config": config_stub,
                },
            ):
                xiaoai_module = importlib.import_module("core.xiaoai")
        finally:
            for module_name, module in saved_modules.items():
                if module is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = module

        calls = []

        class EventManagerStub:
            @staticmethod
            def consume_xiaoai_asr_result(**_kwargs):
                return False

            @staticmethod
            async def wakeup(text, source):
                calls.append((text, source))

        class ConversationStub:
            def __init__(self):
                self.reset_count = 0

            def reset_retries(self):
                self.reset_count += 1

            def apply_runtime_config(self, _config):
                pass

        async def suppress_dialog(dialog_id, reason):
            calls.append(("suppress", dialog_id, reason))

        conversation = ConversationStub()
        xiaoai_module.XiaoAI.conversation = conversation
        xiaoai_module.XiaoAI.refresh_runtime_config()

        line = {
            "header": {
                "namespace": "SpeechRecognizer",
                "name": "RecognizeResult",
                "dialog_id": "dialog-1",
            },
            "payload": {
                "results": [{"text": "你好小黑"}],
                "is_final": True,
                "is_vad_begin": False,
            },
        }
        event = json.dumps(
            {
                "event": "instruction",
                "data": {"NewLine": json.dumps(line, ensure_ascii=False)},
            },
            ensure_ascii=False,
        )

        with (
            mock.patch.object(xiaoai_module, "EventManager", EventManagerStub),
            mock.patch.object(
                xiaoai_module.XiaoAI,
                "_suppress_dialog",
                side_effect=suppress_dialog,
            ),
        ):
            asyncio.run(xiaoai_module.XiaoAI.on_event(event))

        self.assertEqual(conversation.reset_count, 0)
        self.assertEqual(
            calls,
            [
                ("suppress", "dialog-1", "外部唤醒词接管: 你好小黑"),
                ("你好小黑", "kws"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
