import importlib
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class OpenAIHeadersTest(unittest.TestCase):
    def setUp(self):
        sys.modules.setdefault("open_xiaoai_server", types.SimpleNamespace())
        sys.modules.setdefault(
            "aiohttp",
            types.SimpleNamespace(ClientSession=object, ClientTimeout=object),
        )
        sys.modules.setdefault("requests", types.SimpleNamespace(post=None, get=None))
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        sys.modules.pop("core.openai", None)
        self.manager = importlib.import_module("core.openai").OpenAIManager
        self.manager._api_key = ""
        self.manager._session_key = "agent:default:open-xiaoai-bridge"
        self.manager._session_header = ""

    def test_no_session_header_by_default(self):
        """Standard OpenAI compatibility: session_key is not leaked as a header."""
        headers = self.manager._headers()
        self.assertEqual({"Content-Type": "application/json"}, headers)

    def test_session_header_sent_when_configured(self):
        self.manager._session_header = "X-Hermes-Session-Key"
        headers = self.manager._headers()
        self.assertEqual(
            "agent:default:open-xiaoai-bridge",
            headers["X-Hermes-Session-Key"],
        )

    def test_session_header_omitted_when_session_key_empty(self):
        self.manager._session_header = "X-Hermes-Session-Key"
        self.manager._session_key = ""
        self.assertNotIn("X-Hermes-Session-Key", self.manager._headers())

    def test_bearer_auth_and_session_header_coexist(self):
        self.manager._api_key = "secret"
        self.manager._session_header = "X-Hermes-Session-Key"
        headers = self.manager._headers()
        self.assertEqual("Bearer secret", headers["Authorization"])
        self.assertEqual(
            "agent:default:open-xiaoai-bridge",
            headers["X-Hermes-Session-Key"],
        )


if __name__ == "__main__":
    unittest.main()
