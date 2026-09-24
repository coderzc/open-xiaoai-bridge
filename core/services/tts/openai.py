"""OpenAI-compatible text-to-speech client."""

import asyncio
from collections.abc import Mapping
from typing import Any

import aiohttp


PLAYBACK_SUPPORTED_FORMATS = frozenset(("mp3", "flac", "wav", "ogg", "pcm"))


class OpenAITTSError(RuntimeError):
    """The TTS server rejected or returned an invalid response."""


class OpenAITTSTimeoutError(OpenAITTSError):
    """The TTS request exceeded its configured timeout."""


class OpenAITTS:
    """Call a non-streaming OpenAI-compatible ``/v1/audio/speech`` endpoint.

    The client intentionally implements the portable, non-streaming part of
    the OpenAI speech API.  Provider-specific extensions belong in a thin
    subclass (for example :class:`MLXAudioTTS`).
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini-tts"
    DEFAULT_VOICE = "alloy"
    DEFAULT_RESPONSE_FORMAT = "mp3"
    DEFAULT_SPEED = 1.0
    DEFAULT_TIMEOUT = 120.0
    REQUIRES_VOICE = True
    SUPPORTED_RESPONSE_FORMATS = frozenset(
        ("mp3", "opus", "aac", "flac", "wav", "pcm")
    )
    SUPPORTED_STREAM_FORMATS = frozenset(("audio", "sse"))

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        voice: str | Mapping[str, Any] | None = DEFAULT_VOICE,
        instructions: str | None = None,
        response_format: str = DEFAULT_RESPONSE_FORMAT,
        speed: float | None = DEFAULT_SPEED,
        timeout: float = DEFAULT_TIMEOUT,
        extra_body: Mapping[str, Any] | None = None,
        stream_format: str = "audio",
    ):
        self.base_url = str(base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.api_key = str(api_key or "")
        self.model = str(model or self.DEFAULT_MODEL)
        self.voice = self._normalize_voice(voice)
        if self.REQUIRES_VOICE and self.voice is None:
            raise ValueError("voice is required for the OpenAI TTS provider")
        self.instructions = str(instructions) if instructions else None
        self.response_format = str(
            response_format or self.DEFAULT_RESPONSE_FORMAT
        ).strip().lower()
        if self.response_format not in self.SUPPORTED_RESPONSE_FORMATS:
            supported_formats = ", ".join(sorted(self.SUPPORTED_RESPONSE_FORMATS))
            raise ValueError(
                f"response_format must be one of: {supported_formats}"
            )

        self.speed = float(speed) if speed is not None and speed != "" else None
        if self.speed is not None and not 0.25 <= self.speed <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")

        self.timeout = float(timeout)
        self.extra_body = (
            dict(extra_body) if isinstance(extra_body, Mapping) else {}
        )
        extra_stream_format = self.extra_body.get("stream_format")
        if extra_stream_format and str(extra_stream_format).lower() != "audio":
            raise ValueError(
                "stream_format='sse' is not supported by the file playback path"
            )
        self.stream_format = str(stream_format or "audio").strip().lower()
        if self.stream_format not in self.SUPPORTED_STREAM_FORMATS:
            supported_stream_formats = ", ".join(
                sorted(self.SUPPORTED_STREAM_FORMATS)
            )
            raise ValueError(
                "stream_format must be one of: "
                f"{supported_stream_formats}"
            )
        if self.stream_format != "audio":
            raise ValueError(
                "stream_format='sse' is not supported by the file playback path"
            )
        if self.extra_body.get("stream"):
            raise ValueError("OpenAITTS.synthesize does not support stream=true")

    @staticmethod
    def _normalize_voice(
        voice: str | Mapping[str, Any] | None,
    ) -> str | dict[str, Any] | None:
        if isinstance(voice, Mapping):
            normalized = dict(voice)
            if not normalized.get("id"):
                raise ValueError("voice object must include an id")
            return normalized
        return str(voice).strip() if voice else None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | None = None,
        voice_override: str | Mapping[str, Any] | None = None,
    ) -> "OpenAITTS":
        """Build the client from a ``tts.openai`` configuration mapping."""
        config = config if isinstance(config, Mapping) else {}
        configured_voice = config.get("voice", cls.DEFAULT_VOICE)
        voice = voice_override if voice_override is not None else configured_voice
        return cls(
            base_url=config.get("base_url", cls.DEFAULT_BASE_URL),
            api_key=config.get("api_key", ""),
            model=config.get("model", cls.DEFAULT_MODEL),
            voice=voice,
            instructions=config.get("instructions"),
            response_format=config.get(
                "response_format", cls.DEFAULT_RESPONSE_FORMAT
            ),
            speed=config.get("speed", cls.DEFAULT_SPEED),
            timeout=config.get("timeout", cls.DEFAULT_TIMEOUT),
            extra_body=config.get("extra_body"),
            stream_format=config.get("stream_format", "audio"),
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
        if self.instructions is not None:
            payload["instructions"] = self.instructions
        if self.speed is not None:
            payload["speed"] = self.speed
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
                        raise OpenAITTSError(
                            f"HTTP {response.status}: "
                            f"{detail or 'empty error response'}"
                        )
                    if not audio:
                        raise OpenAITTSError(
                            f"HTTP {response.status}: empty audio response"
                        )
                    content_type = getattr(response, "content_type", None)
                    if content_type and not (
                        content_type.startswith("audio/")
                        or content_type
                        in {"application/octet-stream", "binary/octet-stream"}
                    ):
                        raise OpenAITTSError(
                            f"HTTP {response.status}: unexpected content type "
                            f"{content_type}"
                        )
                    return audio
        except asyncio.TimeoutError as exc:
            raise OpenAITTSTimeoutError(
                f"request timed out after {self.timeout:g}s"
            ) from exc
        except aiohttp.ClientError as exc:
            raise OpenAITTSError(f"request failed: {exc}") from exc
