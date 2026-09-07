"""MLX-Audio TTS provider built on the shared OpenAI speech client."""

from collections.abc import Mapping
from typing import Any

from .openai import (
    OpenAITTS,
    OpenAITTSError,
    OpenAITTSTimeoutError,
)


# Keep the old exception names as compatibility aliases for callers that
# imported them from the MLX provider module.
MLXAudioTTSError = OpenAITTSError
MLXAudioTTSTimeoutError = OpenAITTSTimeoutError


class MLXAudioTTS(OpenAITTS):
    """Call MLX-Audio's OpenAI-compatible speech endpoint.

    The HTTP transport and standard fields come from ``OpenAITTS``.  MLX
    extends the request with local-model controls such as ``lang_code`` and
    ``instruct``; its ``instruct`` field is the provider-specific equivalent
    of OpenAI's standard ``instructions`` field.
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
    DEFAULT_MODEL = "Qwen3-TTS"
    DEFAULT_MODE = "custom_voice"
    DEFAULT_RESPONSE_FORMAT = "wav"
    DEFAULT_TIMEOUT = 120.0
    REQUIRES_VOICE = False
    SUPPORTED_MODES = frozenset(("custom_voice", "voice_design"))
    SUPPORTED_RESPONSE_FORMATS = frozenset(
        ("mp3", "wav", "flac", "ogg", "opus")
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
        instructions: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        extra_body: Mapping[str, Any] | None = None,
        stream_format: str = "audio",
    ):
        self.mode = str(mode or self.DEFAULT_MODE).strip().lower()
        if self.mode not in self.SUPPORTED_MODES:
            supported_modes = ", ".join(sorted(self.SUPPORTED_MODES))
            raise ValueError(f"mode must be one of: {supported_modes}")

        self.lang_code = str(lang_code).strip() if lang_code else None
        selected_instruct = instruct or instructions
        self.instruct = str(selected_instruct) if selected_instruct else None
        if self.mode == "voice_design":
            voice = None

        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            voice=voice,
            instructions=None,
            response_format=response_format,
            speed=speed,
            timeout=timeout,
            extra_body=extra_body,
            stream_format=stream_format,
        )

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
            instructions=config.get("instructions"),
            timeout=config.get("timeout", cls.DEFAULT_TIMEOUT),
            extra_body=config.get("extra_body"),
            stream_format=config.get("stream_format", "audio"),
        )

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
