"""MLX-Audio TTS client for its OpenAI-compatible speech endpoint."""

import asyncio
from collections.abc import Mapping
from typing import Any

import aiohttp


class MLXAudioTTSError(RuntimeError):
    """The MLX-Audio TTS server rejected or returned an invalid response."""


class MLXAudioTTSTimeoutError(MLXAudioTTSError):
    """The MLX-Audio TTS request exceeded its configured timeout."""


class MLXAudioTTS:
    """Call MLX-Audio's ``POST /v1/audio/speech`` endpoint."""

    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
    DEFAULT_MODEL = "Qwen3-TTS"
    DEFAULT_MODE = "custom_voice"
    DEFAULT_RESPONSE_FORMAT = "wav"
    DEFAULT_TIMEOUT = 120.0
    SUPPORTED_MODES = frozenset(("custom_voice", "voice_design"))
    SUPPORTED_RESPONSE_FORMATS = frozenset(
        ("wav", "mp3", "flac", "ogg", "pcm")
    )

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        mode: str = DEFAULT_MODE,
        voice: str | None = None,
        lang_code: str | None = None,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        speed: float | None = 1.0,
        instruct: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        extra_body: Mapping[str, Any] | None = None,
    ):
        self.base_url = str(base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = str(api_key or "")
        self.model = str(model or self.DEFAULT_MODEL)
        self.mode = str(mode or self.DEFAULT_MODE).strip().lower()
        if self.mode not in self.SUPPORTED_MODES:
            supported_modes = ", ".join(sorted(self.SUPPORTED_MODES))
            raise ValueError(f"mode must be one of: {supported_modes}")
        self.voice = str(voice).strip() if voice else None
        if self.mode == "voice_design":
            self.voice = None
        self.lang_code = str(lang_code).strip() if lang_code else None
        self.response_format = str(
            response_format or self.DEFAULT_RESPONSE_FORMAT
        ).strip().lower()
        if self.response_format not in self.SUPPORTED_RESPONSE_FORMATS:
            supported_formats = ", ".join(sorted(self.SUPPORTED_RESPONSE_FORMATS))
            raise ValueError(f"response_format must be one of: {supported_formats}")
        self.speed = float(speed) if speed is not None and speed != "" else None
        self.instruct = str(instruct) if instruct else None
        self.timeout = float(timeout)
        self.extra_body = dict(extra_body) if isinstance(extra_body, Mapping) else {}
        if self.extra_body.get("stream"):
            raise ValueError("MLXAudioTTS.synthesize does not support stream=true")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        voice_override: str | None = None,
    ) -> "MLXAudioTTS":
        """Build the adapter from ``tts.mlx_audio`` settings."""
        config = config if isinstance(config, Mapping) else {}
        mode = config.get("mode", cls.DEFAULT_MODE)
        configured_voice = config.get("voice")
        voice = voice_override or configured_voice
        return cls(
            base_url=config.get("base_url", cls.DEFAULT_BASE_URL),
            api_key=config.get("api_key", ""),
            model=config.get("model", cls.DEFAULT_MODEL),
            mode=mode,
            voice=voice,
            lang_code=config.get("lang_code"),
            response_format=config.get(
                "response_format", cls.DEFAULT_RESPONSE_FORMAT
            ),
            speed=config.get("speed", 1.0),
            instruct=config.get("instruct"),
            timeout=config.get("timeout", cls.DEFAULT_TIMEOUT),
            extra_body=config.get("extra_body"),
        )

    @property
    def speech_url(self) -> str:
        """Return the configured base URL with the speech path appended once."""
        if self.base_url.endswith("/audio/speech"):
            return self.base_url
        return f"{self.base_url}/audio/speech"

    def _payload(self, text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        payload: dict[str, Any] = dict(self.extra_body)
        payload.update(
            {
                "model": self.model,
                "input": text,
                "response_format": self.response_format,
            }
        )
        if self.voice is not None:
            payload["voice"] = self.voice
        if self.lang_code is not None:
            payload["lang_code"] = self.lang_code
        if self.speed is not None:
            payload["speed"] = self.speed
        if self.instruct is not None:
            payload["instruct"] = self.instruct
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def synthesize(self, text: str) -> bytes:
        """Synthesize ``text`` and return the encoded audio response."""
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.speech_url,
                    json=self._payload(text),
                    headers=self._headers(),
                ) as response:
                    audio = await response.read()
                    if response.status < 200 or response.status >= 300:
                        detail = audio[:500].decode("utf-8", errors="replace")
                        raise MLXAudioTTSError(
                            f"HTTP {response.status}: {detail or 'empty error response'}"
                        )
                    if not audio:
                        raise MLXAudioTTSError(
                            f"HTTP {response.status}: empty audio response"
                        )
                    content_type = getattr(response, "content_type", None)
                    if content_type and not (
                        content_type.startswith("audio/")
                        or content_type
                        in {"application/octet-stream", "binary/octet-stream"}
                    ):
                        raise MLXAudioTTSError(
                            f"HTTP {response.status}: unexpected content type "
                            f"{content_type}"
                        )
                    return audio
        except asyncio.TimeoutError as exc:
            raise MLXAudioTTSTimeoutError(
                f"request timed out after {self.timeout:g}s"
            ) from exc
        except aiohttp.ClientError as exc:
            raise MLXAudioTTSError(f"request failed: {exc}") from exc
