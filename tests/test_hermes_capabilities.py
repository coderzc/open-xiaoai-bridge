import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class _Response:
    def __init__(self, status, body=None, enter_error=None):
        self.status = status
        self.body = body
        self.headers = {}
        self.enter_error = enter_error

    async def __aenter__(self):
        if self.enter_error:
            raise self.enter_error
        return self

    async def __aexit__(self, *_args):
        return False

    async def json(self, **_kwargs):
        return self.body


class _Session:
    def __init__(self, get_response, post_response=None, calls=None, **_kwargs):
        self.get_response = get_response
        self.post_response = post_response
        self.calls = calls if calls is not None else []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.get_response

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.post_response


class HermesCapabilitiesTest(unittest.TestCase):
    def setUp(self):
        sys.modules["open_xiaoai_server"] = types.SimpleNamespace()
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        sys.modules.pop("core.hermes", None)
        self.module = importlib.import_module("core.hermes")
        self.hermes = self.module.HermesManager
        self.hermes._initialized = True
        self.hermes._enabled = True
        self.hermes._base_url = "http://hermes.test/v1"
        self.hermes._api_key = "secret"
        self.hermes._profile = ""
        self.hermes._profile_api_keys = {}
        self.hermes._capabilities_checked = False
        self.hermes._capabilities = None
        self.hermes._capabilities_diagnostic = None
        self.hermes._sessions.clear()
        self.hermes._hermes_session_ids.clear()

    def _run_probe(self, response, calls=None):
        calls = calls if calls is not None else []

        def session_factory(**kwargs):
            return _Session(response, calls=calls, **kwargs)

        with (
            mock.patch.object(
                self.module.aiohttp,
                "ClientSession",
                side_effect=session_factory,
            ),
            mock.patch.object(
                self.module.aiohttp,
                "ClientTimeout",
                return_value=object(),
            ) as timeout,
        ):
            result = asyncio.run(self.hermes.connect())
        return result, calls, timeout

    def test_capabilities_success_is_recorded_and_only_requested_once(self):
        body = {
            "object": "hermes.api_server.capabilities",
            "platform": "hermes-agent",
            "features": {"chat_completions_streaming": True},
            "endpoints": {
                "chat_completions": {"path": "/v1/chat/completions"},
                "skills": {"path": "/v1/skills"},
            },
        }
        calls = []
        response = _Response(200, body)

        def session_factory(**kwargs):
            return _Session(response, calls=calls, **kwargs)

        with (
            mock.patch.object(
                self.module.aiohttp,
                "ClientSession",
                side_effect=session_factory,
            ),
            mock.patch.object(
                self.module.aiohttp,
                "ClientTimeout",
                return_value=object(),
            ),
        ):
            self.assertTrue(asyncio.run(self.hermes.connect()))
            self.assertTrue(asyncio.run(self.hermes.connect()))

        self.assertEqual(1, len(calls))
        self.assertEqual(
            "http://hermes.test/v1/capabilities",
            calls[0][1],
        )
        self.assertEqual(
            "Bearer secret",
            calls[0][2]["headers"]["Authorization"],
        )
        self.assertEqual(body, self.hermes.get_capabilities())

    def test_capabilities_probe_is_independent_of_response_mode(self):
        body = {
            "object": "hermes.api_server.capabilities",
            "platform": "hermes-agent",
            "endpoints": {},
        }
        for mode in ("streaming", "complete"):
            with self.subTest(mode=mode):
                self.hermes._response_mode = mode
                self.hermes._capabilities_checked = False
                self.hermes._capabilities = None
                result, calls, _timeout = self._run_probe(
                    _Response(200, body)
                )
                self.assertTrue(result)
                self.assertEqual(1, len(calls))
                self.assertEqual(
                    "http://hermes.test/v1/capabilities",
                    calls[0][1],
                )

    def test_capabilities_404_is_non_blocking(self):
        result, _calls, _timeout = self._run_probe(_Response(404))
        self.assertTrue(result)
        self.assertIsNone(self.hermes.get_capabilities())
        self.assertEqual(
            "unsupported (HTTP 404)",
            self.hermes._capabilities_diagnostic,
        )

    def test_capabilities_timeout_is_non_blocking(self):
        response = _Response(200, enter_error=asyncio.TimeoutError())
        result, _calls, timeout = self._run_probe(response)
        self.assertTrue(result)
        timeout.assert_called_once_with(
            total=self.hermes.CAPABILITIES_TIMEOUT
        )
        self.assertEqual("timeout", self.hermes._capabilities_diagnostic)

    def test_malformed_capabilities_is_non_blocking(self):
        result, _calls, _timeout = self._run_probe(
            _Response(200, {"object": "not-hermes"})
        )
        self.assertTrue(result)
        self.assertIsNone(self.hermes.get_capabilities())
        self.assertEqual(
            "malformed response",
            self.hermes._capabilities_diagnostic,
        )

    def test_old_server_can_chat_after_capabilities_404(self):
        calls = []
        get_response = _Response(404)
        post_response = _Response(
            200,
            {
                "choices": [
                    {"message": {"content": "兼容回答"}}
                ]
            },
        )

        def session_factory(**kwargs):
            return _Session(
                get_response,
                post_response=post_response,
                calls=calls,
                **kwargs,
            )

        with (
            mock.patch.object(
                self.module.aiohttp,
                "ClientSession",
                side_effect=session_factory,
            ),
            mock.patch.object(
                self.module.aiohttp,
                "ClientTimeout",
                return_value=object(),
            ),
        ):
            self.assertTrue(asyncio.run(self.hermes.connect()))
            reply = asyncio.run(
                self.hermes._request_chat_completion("问题")
            )

        self.assertEqual("兼容回答", reply)
        self.assertEqual(["GET", "POST"], [call[0] for call in calls])


if __name__ == "__main__":
    unittest.main()
