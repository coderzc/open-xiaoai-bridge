import asyncio
import importlib
import sys
import types
import unittest
from unittest.mock import patch


ROOT = importlib.import_module("pathlib").Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


tts_module = importlib.import_module("core.services.tts.mlx_audio")
MLXAudioTTS = tts_module.MLXAudioTTS
MLXAudioTTSTimeoutError = tts_module.MLXAudioTTSTimeoutError


class FakeResponse:
    def __init__(
        self,
        status=200,
        body=b"RIFF-audio",
        read_error=None,
        content_type="audio/wav",
    ):
        self.status = status
        self.body = body
        self.read_error = read_error
        self.content_type = content_type

    async def read(self):
        if self.read_error:
            raise self.read_error
        return self.body


class ResponseContext:
    def __init__(self, response=None, enter_error=None):
        self.response = response
        self.enter_error = enter_error

    async def __aenter__(self):
        if self.enter_error:
            raise self.enter_error
        return self.response

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class FakeSession:
    def __init__(self, response=None, enter_error=None):
        self.response = response
        self.enter_error = enter_error
        self.url = None
        self.request_kwargs = None

    async def __aenter__(self):
        if self.enter_error:
            raise self.enter_error
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def post(self, url, **kwargs):
        self.url = url
        self.request_kwargs = kwargs
        return ResponseContext(self.response, self.enter_error)


class MLXAudioTTSTest(unittest.IsolatedAsyncioTestCase):
    async def test_voice_design_posts_instruction_without_preset_voice(self):
        session = FakeSession(FakeResponse())
        tts = MLXAudioTTS(
            base_url="http://mlx-audio:8000/v1",
            model="mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
            mode="voice_design",
            voice="Dylan",
            lang_code="Chinese",
            response_format="wav",
            instruct="年轻中文男声，标准普通话，无明显地域口音。",
        )

        with patch.object(tts_module.aiohttp, "ClientSession", return_value=session):
            with patch.object(tts_module.aiohttp, "ClientTimeout", side_effect=lambda total: total):
                await tts.synthesize("你好，这是一次音色测试。")

        self.assertEqual(
            {
                "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-6bit",
                "input": "你好，这是一次音色测试。",
                "lang_code": "Chinese",
                "response_format": "wav",
                "speed": 1.0,
                "instruct": "年轻中文男声，标准普通话，无明显地域口音。",
            },
            session.request_kwargs["json"],
        )

    async def test_synthesize_posts_mlx_audio_speech_payload(self):
        session = FakeSession(FakeResponse())
        tts = MLXAudioTTS(
            base_url="http://mlx-audio:8000/v1",
            api_key="secret",
            model="Qwen3-TTS",
            voice="Vivian",
            lang_code="Chinese",
            response_format="wav",
            speed=1.25,
            instruct="用温柔的语气",
            timeout=7,
            extra_body={"custom_field": "enabled"},
        )

        with patch.object(tts_module.aiohttp, "ClientSession", return_value=session):
            with patch.object(tts_module.aiohttp, "ClientTimeout", side_effect=lambda total: total):
                audio = await tts.synthesize("你好")

        self.assertEqual(b"RIFF-audio", audio)
        self.assertEqual("http://mlx-audio:8000/v1/audio/speech", session.url)
        self.assertEqual(
            {
                "model": "Qwen3-TTS",
                "input": "你好",
                "voice": "Vivian",
                "lang_code": "Chinese",
                "response_format": "wav",
                "speed": 1.25,
                "instruct": "用温柔的语气",
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

    async def test_synthesize_raises_clear_error_for_non_success_response(self):
        session = FakeSession(FakeResponse(status=503, body=b"service unavailable"))
        tts = MLXAudioTTS(base_url="http://mlx-audio:8000/v1")

        with patch.object(tts_module.aiohttp, "ClientSession", return_value=session):
            with patch.object(tts_module.aiohttp, "ClientTimeout", side_effect=lambda total: total):
                with self.assertRaisesRegex(RuntimeError, "HTTP 503.*service unavailable"):
                    await tts.synthesize("你好")

    async def test_synthesize_converts_request_timeout(self):
        session = FakeSession(enter_error=asyncio.TimeoutError())
        tts = MLXAudioTTS(base_url="http://mlx-audio:8000/v1", timeout=3)

        with patch.object(tts_module.aiohttp, "ClientSession", return_value=session):
            with patch.object(tts_module.aiohttp, "ClientTimeout", side_effect=lambda total: total):
                with self.assertRaises(MLXAudioTTSTimeoutError):
                    await tts.synthesize("你好")

    async def test_synthesize_rejects_successful_non_audio_response(self):
        session = FakeSession(
            FakeResponse(body=b'{"error":"not audio"}', content_type="application/json")
        )
        tts = MLXAudioTTS(base_url="http://mlx-audio:8000/v1")

        with patch.object(tts_module.aiohttp, "ClientSession", return_value=session):
            with patch.object(tts_module.aiohttp, "ClientTimeout", side_effect=lambda total: total):
                with self.assertRaisesRegex(RuntimeError, "unexpected content type"):
                    await tts.synthesize("你好")

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            MLXAudioTTS(mode="voice-design")

    def test_streaming_config_is_rejected_by_file_adapter(self):
        with self.assertRaisesRegex(ValueError, "does not support stream=true"):
            MLXAudioTTS(extra_body={"stream": True})

    def test_output_format_must_be_supported_by_bridge_decoder(self):
        with self.assertRaisesRegex(ValueError, "response_format must be one of"):
            MLXAudioTTS(response_format="opus")


class MLXAudioTTSFallbackTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.setdefault("open_xiaoai_server", types.SimpleNamespace())
        sys.modules.setdefault(
            "aiohttp",
            types.SimpleNamespace(ClientSession=object, ClientTimeout=object),
        )
        cls.openai_module = importlib.import_module("core.openai")
        cls.manager = cls.openai_module.OpenAIManager
        cls.config_manager = importlib.import_module("core.utils.config").ConfigManager
        cls.adapter = MLXAudioTTS

    def setUp(self):
        self.previous_provider = self.manager._tts_provider
        self.previous_speaker = self.manager._tts_speaker
        self.previous_session_speakers = self.manager._session_tts_speakers
        self.manager._tts_provider = "mlx_audio"
        self.manager._tts_speaker = None
        self.manager._session_tts_speakers = {}

    def tearDown(self):
        self.manager._tts_provider = self.previous_provider
        self.manager._tts_speaker = self.previous_speaker
        self.manager._session_tts_speakers = self.previous_session_speakers

    async def test_success_plays_server_file_and_removes_temporary_audio(self):
        speaker = unittest.mock.MagicMock()
        speaker.play_server_file = unittest.mock.AsyncMock(return_value=True)
        speaker.play = unittest.mock.AsyncMock()
        config = {
            "base_url": "http://mlx-audio:8000/v1",
            "model": "Qwen3-TTS",
            "voice": "Vivian",
            "response_format": "wav",
        }

        with patch.object(
            self.config_manager.instance(), "get_app_config", return_value=config
        ):
            with patch.object(
                self.adapter, "synthesize", new=unittest.mock.AsyncMock(return_value=b"RIFF-audio")
            ):
                with patch.object(
                    importlib.import_module("core.ref"), "get_speaker", return_value=speaker
                ):
                    await self.manager._play_response_with_tts(
                        "你好", tts_speaker="Vivian"
                    )

        speaker.play_server_file.assert_awaited_once()
        audio_path = speaker.play_server_file.await_args.kwargs["file_path"]
        self.assertFalse(importlib.import_module("os").path.exists(audio_path))
        speaker.play.assert_not_awaited()

    async def test_playback_error_falls_back_and_still_removes_temporary_audio(self):
        speaker = unittest.mock.MagicMock()
        speaker.play_server_file = unittest.mock.AsyncMock(
            side_effect=RuntimeError("rust playback failed")
        )
        speaker.play = unittest.mock.AsyncMock(return_value=True)
        config = {"base_url": "http://mlx-audio:8000/v1", "response_format": "wav"}

        with patch.object(
            self.config_manager.instance(), "get_app_config", return_value=config
        ):
            with patch.object(
                self.adapter, "synthesize", new=unittest.mock.AsyncMock(return_value=b"RIFF-audio")
            ):
                with patch.object(
                    importlib.import_module("core.ref"), "get_speaker", return_value=speaker
                ):
                    await self.manager._play_response_with_tts("你好")

        audio_path = speaker.play_server_file.await_args.kwargs["file_path"]
        self.assertFalse(importlib.import_module("os").path.exists(audio_path))
        speaker.play.assert_awaited_once_with(text="你好", blocking=True)

    async def test_tts_timeout_falls_back_to_xiaoai_native_tts(self):
        speaker = unittest.mock.MagicMock()
        speaker.play = unittest.mock.AsyncMock(return_value=True)
        config = {"base_url": "http://mlx-audio:8000/v1"}

        with patch.object(
            self.config_manager.instance(), "get_app_config", return_value=config
        ):
            with patch.object(
                self.adapter,
                "synthesize",
                new=unittest.mock.AsyncMock(
                    side_effect=MLXAudioTTSTimeoutError("request timed out")
                ),
            ):
                with patch.object(
                    importlib.import_module("core.ref"), "get_speaker", return_value=speaker
                ):
                    await self.manager._play_response_with_tts("你好")

        speaker.play.assert_awaited_once_with(text="你好", blocking=True)

    def test_missing_provider_keeps_legacy_xiaoai_and_doubao_selection(self):
        self.manager._tts_provider = None
        self.assertEqual(
            "xiaoai", self.manager._resolve_tts_provider("xiaoai")
        )
        self.assertEqual("doubao", self.manager._resolve_tts_provider(None))


if __name__ == "__main__":
    unittest.main()
