import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class ExternalConversationHookTest(unittest.TestCase):
    def setUp(self):
        sys.modules["open_xiaoai_server"] = types.SimpleNamespace(
            decode_audio=lambda *_args, **_kwargs: b"",
        )
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        sys.modules.pop("core.external_conversation", None)
        module = importlib.import_module("core.external_conversation")
        self.controller = module.ExternalConversationController.__new__(
            module.ExternalConversationController
        )
        self.controller._play_send_sound = mock.AsyncMock()

    def test_local_asr_default_path_preserves_tracking_order(self):
        calls = []
        backend = types.SimpleNamespace(
            _rule_prompt="原有规则",
            _send_and_track=mock.AsyncMock(
                side_effect=lambda text: calls.append(
                    ("send", text)
                ) or "run-1"
            ),
            _wait_response=mock.AsyncMock(
                side_effect=lambda run_id: calls.append(
                    ("wait", run_id)
                ) or "回答"
            ),
        )
        self.controller.backend = backend
        self.controller._play_send_sound = mock.AsyncMock(
            side_effect=lambda: calls.append(("sound", None))
        )

        response = asyncio.run(
            self.controller._request_backend_turn(
                "问题",
                play_send_sound=True,
            )
        )

        self.assertEqual(("回答", False), response)
        self.assertEqual(
            [
                ("send", "问题\n原有规则"),
                ("sound", None),
                ("wait", "run-1"),
            ],
            calls,
        )

    def test_xiaoai_asr_default_path_stays_non_streaming(self):
        backend = types.SimpleNamespace(
            _rule_prompt="",
            send=mock.AsyncMock(return_value="整段回答"),
        )
        self.controller.backend = backend

        response = asyncio.run(
            self.controller._request_backend_turn(
                "问题",
                play_send_sound=False,
            )
        )

        self.assertEqual(("整段回答", False), response)
        backend.send.assert_awaited_once_with(
            "问题",
            wait_response=True,
        )
        self.controller._play_send_sound.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
