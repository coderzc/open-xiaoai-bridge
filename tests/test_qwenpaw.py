import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


class QwenPawManagerTest(unittest.TestCase):
    def setUp(self):
        self.server_stub = types.SimpleNamespace()
        sys.modules.setdefault("open_xiaoai_server", self.server_stub)
        sys.modules.setdefault(
            "aiohttp",
            types.SimpleNamespace(ClientSession=object, ClientTimeout=object),
        )
        sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
        sys.modules.pop("core.qwenpaw", None)
        self.qwenpaw_module = importlib.import_module("core.qwenpaw")
        self.manager = self.qwenpaw_module.QwenPawManager

    def test_build_payload_uses_qwenpaw_agent_request_shape(self):
        self.manager._session_key = "speaker-session"
        self.manager._user_id = "speaker-user"

        payload = self.manager._build_payload("你好")

        self.assertEqual("speaker-session", payload["session_id"])
        self.assertEqual("speaker-user", payload["user_id"])
        self.assertTrue(payload["stream"])
        self.assertEqual(
            [{"role": "user", "content": [{"type": "text", "text": "你好"}]}],
            payload["input"],
        )

    def test_extract_response_text_from_console_task_result(self):
        body = {
            "status": "completed",
            "session_id": "speaker-session",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "你好，"},
                        {"type": "text", "text": "我是小爪"},
                    ],
                }
            ],
        }

        self.assertEqual("你好，我是小爪", self.manager._extract_response_text(body))

    def test_extract_response_text_ignores_user_messages_and_usage(self):
        body = {
            "output": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "不要回显我"}],
                },
                {"type": "turn_usage", "text": "usage"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "只读助手回复"}],
                },
            ]
        }

        self.assertEqual("只读助手回复", self.manager._extract_response_text(body))

    def test_headers_include_agent_id(self):
        self.manager._agent_id = "assistant"
        self.manager._auth_token = ""
        self.manager._auth_header = "Authorization"
        self.manager._auth_scheme = "Bearer"
        self.manager._extra_headers = {}

        self.assertEqual(
            {"Content-Type": "application/json", "X-Agent-Id": "assistant"},
            self.manager._headers(),
        )

    def test_headers_include_bearer_auth_when_token_configured(self):
        self.manager._agent_id = "assistant"
        self.manager._auth_token = "secret-token"
        self.manager._auth_header = "Authorization"
        self.manager._auth_scheme = "Bearer"
        self.manager._extra_headers = {}

        self.assertEqual(
            {
                "Content-Type": "application/json",
                "X-Agent-Id": "assistant",
                "Authorization": f"{self.manager._auth_scheme} {self.manager._auth_token}",
            },
            self.manager._headers(),
        )

    def test_headers_support_custom_auth_header_without_scheme(self):
        self.manager._agent_id = "assistant"
        self.manager._auth_token = "secret-token"
        self.manager._auth_header = "X-API-Key"
        self.manager._auth_scheme = ""
        self.manager._extra_headers = {}

        self.assertEqual(
            {
                "Content-Type": "application/json",
                "X-Agent-Id": "assistant",
                "X-API-Key": "secret-token",
            },
            self.manager._headers(),
        )

    def test_headers_merge_extra_headers(self):
        self.manager._agent_id = "assistant"
        self.manager._auth_token = ""
        self.manager._auth_header = "Authorization"
        self.manager._auth_scheme = "Bearer"
        self.manager._extra_headers = {"X-Custom": "custom-value"}

        self.assertEqual(
            {
                "Content-Type": "application/json",
                "X-Agent-Id": "assistant",
                "X-Custom": "custom-value",
            },
            self.manager._headers(),
        )


class WakeupKeywordStartupQwenPawTest(unittest.TestCase):
    def test_keyword_generation_enabled_for_qwenpaw(self):
        spec = importlib.util.spec_from_file_location(
            "kws_keywords_for_qwenpaw_test",
            ROOT / "core/services/audio/kws/keywords.py",
        )
        keywords = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(keywords)

        with mock.patch.dict(
            "os.environ",
            {
                "XIAOZHI_ENABLE": "",
                "OPENCLAW_ENABLE": "",
                "OPENCLAW_ENABLED": "",
                "OPENAI_ENABLE": "",
                "QWENPAW_ENABLE": "1",
            },
            clear=False,
        ):
            should_run, reason = keywords.should_generate_keywords()

        self.assertTrue(should_run)
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
