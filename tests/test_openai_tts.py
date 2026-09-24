import importlib
import sys
import types
import unittest
from unittest.mock import patch


ROOT = importlib.import_module("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


tts_module = importlib.import_module("core.services.tts.openai")
OpenAITTS = tts_module.OpenAITTS
sys.modules.setdefault("open_xiaoai_server", types.SimpleNamespace())


class FakeResponse:
    def __init__(self, status=200, body=b"audio", content_type="audio/mpeg"):
        self.status = status
        self.body = body
        self.content_type = content_type

    async def read(self):
        return self.body


class ResponseContext:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.url = None
        self.request_kwargs = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, **kwargs):
        self.url = url
        self.request_kwargs = kwargs
        return ResponseContext(self.response)


class OpenAITTSTest(unittest.IsolatedAsyncioTestCase):
    async def test_posts_standard_openai_speech_payload(self):
        session = FakeSession(FakeResponse())
        tts = OpenAITTS(
            base_url="http://tts.example/v1",
            api_key="secret",
            model="gpt-4o-mini-tts",
            voice={"id": "voice_123"},
            instructions="自然、放松地说话",
            response_format="opus",
            speed=1.25,
            extra_body={"custom_field": "enabled"},
        )

        with patch.object(tts_module.aiohttp, "ClientSession", return_value=session):
            with patch.object(
                tts_module.aiohttp,
                "ClientTimeout",
                side_effect=lambda total: total,
            ):
                audio = await tts.synthesize("你好")

        self.assertEqual(b"audio", audio)
        self.assertEqual("http://tts.example/v1/audio/speech", session.url)
        self.assertEqual(
            {
                "model": "gpt-4o-mini-tts",
                "input": "你好",
                "voice": {"id": "voice_123"},
                "instructions": "自然、放松地说话",
                "response_format": "opus",
                "speed": 1.25,
                "custom_field": "enabled",
            },
            session.request_kwargs["json"],
        )
        self.assertEqual(
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer secret",
            },
            session.request_kwargs["headers"],
        )

    def test_supports_official_non_streaming_formats(self):
        for response_format in ("mp3", "opus", "aac", "flac", "wav", "pcm"):
            self.assertEqual(
                response_format,
                OpenAITTS(response_format=response_format).response_format,
            )

    def test_rejects_sse_streaming(self):
        with self.assertRaisesRegex(ValueError, "stream_format='sse'"):
            OpenAITTS(stream_format="sse")

    def test_rejects_stream_body_extension(self):
        with self.assertRaisesRegex(ValueError, "stream=true"):
            OpenAITTS(extra_body={"stream": True})

    def test_rejects_sse_stream_body_extension(self):
        with self.assertRaisesRegex(ValueError, "stream_format='sse'"):
            OpenAITTS(extra_body={"stream_format": "sse"})

    def test_rejects_speed_outside_official_range(self):
        with self.assertRaisesRegex(ValueError, "between 0.25 and 4.0"):
            OpenAITTS(speed=4.1)

    def test_from_config_allows_voice_override(self):
        tts = OpenAITTS.from_config(
            {
                "base_url": "http://tts.example/v1",
                "voice": "alloy",
            },
            voice_override="sage",
        )
        self.assertEqual("sage", tts.voice)

    def test_rejects_custom_voice_without_id(self):
        with self.assertRaisesRegex(ValueError, "must include an id"):
            OpenAITTS(voice={"name": "custom"})


class OpenAITTSPlaybackTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.openai_module = importlib.import_module("core.openai")
        cls.manager = cls.openai_module.OpenAIManager
        cls.config_manager = importlib.import_module(
            "core.utils.config"
        ).ConfigManager

    def setUp(self):
        self.previous_provider = self.manager._tts_provider
        self.previous_speaker = self.manager._tts_speaker
        self.previous_session_speakers = self.manager._session_tts_speakers
        self.manager._tts_provider = "openai"
        self.manager._tts_speaker = None
        self.manager._session_tts_speakers = {}

    def tearDown(self):
        self.manager._tts_provider = self.previous_provider
        self.manager._tts_speaker = self.previous_speaker
        self.manager._session_tts_speakers = self.previous_session_speakers

    async def test_openai_provider_uses_shared_file_playback(self):
        speaker = unittest.mock.MagicMock()
        speaker.play_server_file = unittest.mock.AsyncMock(return_value=True)
        speaker.play = unittest.mock.AsyncMock()
        config = {
            "base_url": "http://tts.example/v1",
            "model": "gpt-4o-mini-tts",
            "voice": "alloy",
            "response_format": "wav",
        }

        with patch.object(
            self.config_manager.instance(), "get_app_config", return_value=config
        ):
            with patch.object(
                OpenAITTS,
                "synthesize",
                new=unittest.mock.AsyncMock(return_value=b"RIFF-audio"),
            ):
                with patch.object(
                    importlib.import_module("core.ref"),
                    "get_speaker",
                    return_value=speaker,
                ):
                    await self.manager._play_response_with_tts(
                        "你好", tts_speaker="sage"
                    )

        speaker.play_server_file.assert_awaited_once()
        audio_path = speaker.play_server_file.await_args.kwargs["file_path"]
        self.assertFalse(importlib.import_module("os").path.exists(audio_path))
        speaker.play.assert_not_awaited()

    async def test_unsupported_playback_format_falls_back_before_request(self):
        speaker = unittest.mock.MagicMock()
        speaker.play = unittest.mock.AsyncMock(return_value=True)
        config = {
            "base_url": "http://tts.example/v1",
            "response_format": "opus",
        }
        synthesize = unittest.mock.AsyncMock(return_value=b"audio")

        with patch.object(
            self.config_manager.instance(), "get_app_config", return_value=config
        ):
            with patch.object(OpenAITTS, "synthesize", new=synthesize):
                with patch.object(
                    importlib.import_module("core.ref"),
                    "get_speaker",
                    return_value=speaker,
                ):
                    await self.manager._play_response_with_tts(
                        "你好", tts_speaker="alloy"
                    )

        synthesize.assert_not_awaited()
        speaker.play.assert_awaited_once_with(text="你好", blocking=True)


if __name__ == "__main__":
    unittest.main()
