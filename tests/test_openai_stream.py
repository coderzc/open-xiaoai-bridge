import asyncio
import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


class OpenAIStreamHelpersTest(unittest.TestCase):
    def setUp(self):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        self.module = importlib.import_module("core.openai_stream")

    def test_sse_decoder_handles_network_and_utf8_boundaries(self):
        decoder = self.module.SSEDecoder()
        payload = (
            'event: custom.progress\n'
            'data: {"status":"running"}\n\n'
            'data: {"choices":[{"delta":{"content":"你好。"},'
            '"finish_reason":null}]}\n\n'
        )
        encoded = payload.encode("utf-8")
        utf8_decoder = __import__("codecs").getincrementaldecoder("utf-8")()
        events = []
        for chunk in (encoded[:13], encoded[13:61], encoded[61:77], encoded[77:]):
            events.extend(decoder.feed(utf8_decoder.decode(chunk)))
        events.extend(
            decoder.feed(
                utf8_decoder.decode(b"", final=True),
                final=True,
            )
        )

        self.assertEqual(
            ["custom.progress", "message"],
            [event.event for event in events],
        )
        delta, finish_reason = self.module.extract_openai_delta(
            events[1].data
        )
        self.assertEqual("你好。", delta)
        self.assertIsNone(finish_reason)

    def test_sentence_chunker_flushes_and_bounds_long_text(self):
        chunker = self.module.SentenceChunker(
            min_chars=4,
            max_chars=8,
        )

        chunks = chunker.feed("第一句话。第二句话还没有结束")
        chunks.extend(chunker.feed("。尾巴", final=True))

        self.assertEqual("第一句话。", chunks[0])
        self.assertEqual(
            "第一句话。第二句话还没有结束。尾巴",
            "".join(chunks),
        )
        self.assertTrue(all(len(chunk) <= 8 for chunk in chunks))

    def test_transport_forwards_named_events_without_interpreting_them(self):
        payload = (
            "event: vendor.progress\n"
            'data: {"internal":"opaque"}\n\n'
            'data: {"choices":[{"delta":{"content":"最终答案。"},'
            '"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        chunks = [payload[:41], payload[41:83], payload[83:]]

        class FakeContent:
            async def iter_any(self):
                for chunk in chunks:
                    yield chunk

        class FakeResponse:
            status = 200
            headers = {"Content-Type": "text/event-stream"}
            content = FakeContent()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        class FakeSession:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def post(self, *_args, **_kwargs):
                return FakeResponse()

        manager = types.SimpleNamespace(
            _initialized=True,
            _enabled=True,
            _session_key="agent:test",
            _sessions={},
            _model="model",
            _extra_body={},
            _temperature=None,
            _max_tokens=None,
            _timeout=10,
            _build_messages=lambda history, text: [
                {"role": "user", "content": text}
            ],
            _chat_completions_url=lambda: "http://example.test/v1/chat/completions",
            _headers=lambda: {"Content-Type": "application/json"},
            _capture_response_headers=mock.Mock(),
            _append_history=lambda history, text, response: history.extend(
                [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": response},
                ]
            ),
        )
        deltas = []
        events = []

        async def scenario():
            with (
                mock.patch.object(
                    self.module.aiohttp,
                    "ClientSession",
                    FakeSession,
                ),
                mock.patch.object(
                    self.module.aiohttp,
                    "ClientTimeout",
                    lambda **_kwargs: object(),
                ),
                mock.patch.object(self.module.logger, "ai_response"),
            ):
                return await self.module.stream_openai_chat_completion(
                    manager,
                    "问题",
                    on_delta=lambda value: deltas.append(value),
                    on_event=lambda event: events.append(event),
                    log_name="Test Stream",
                )

        self.assertEqual("最终答案。", asyncio.run(scenario()))
        self.assertEqual(["最终答案。"], deltas)
        self.assertEqual(["vendor.progress"], [event.event for event in events])
        self.assertEqual('{"internal":"opaque"}', events[0].data)
        manager._capture_response_headers.assert_called_once_with(
            FakeResponse.headers,
            session_key="agent:test",
        )

    def test_interrupted_stream_does_not_write_partial_history(self):
        payload = (
            'data: {"choices":[{"delta":{"content":"部分回答"},'
            '"finish_reason":null}]}\n\n'
        ).encode()

        class FakeContent:
            async def iter_any(self):
                yield payload
                raise RuntimeError("connection lost")

        class FakeResponse:
            status = 200
            headers = {"Content-Type": "text/event-stream"}
            content = FakeContent()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        class FakeSession:
            def __init__(self, **_kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def post(self, *_args, **_kwargs):
                return FakeResponse()

        manager = types.SimpleNamespace(
            _initialized=True,
            _enabled=True,
            _session_key="agent:test",
            _sessions={},
            _model="model",
            _extra_body={},
            _temperature=None,
            _max_tokens=None,
            _timeout=10,
            _build_messages=lambda history, text: [
                {"role": "user", "content": text}
            ],
            _chat_completions_url=lambda: (
                "http://example.test/v1/chat/completions"
            ),
            _headers=lambda: {"Content-Type": "application/json"},
            _capture_response_headers=mock.Mock(),
            _append_history=mock.Mock(),
        )
        deltas = []

        async def scenario():
            with (
                mock.patch.object(
                    self.module.aiohttp,
                    "ClientSession",
                    FakeSession,
                ),
                mock.patch.object(
                    self.module.aiohttp,
                    "ClientTimeout",
                    lambda **_kwargs: object(),
                ),
            ):
                await self.module.stream_openai_chat_completion(
                    manager,
                    "问题",
                    on_delta=lambda value: deltas.append(value),
                )

        with self.assertRaisesRegex(RuntimeError, "connection lost"):
            asyncio.run(scenario())

        self.assertEqual(["部分回答"], deltas)
        manager._append_history.assert_not_called()
        self.assertEqual({"agent:test": []}, manager._sessions)


if __name__ == "__main__":
    unittest.main()
