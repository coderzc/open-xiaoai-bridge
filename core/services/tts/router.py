"""Shared TTS provider routing and playback for all conversation backends."""

import os
import tempfile

import open_xiaoai_server

from core.services.tts.doubao import DoubaoTTS
from core.services.tts.mlx_audio import MLXAudioTTS
from core.services.tts.openai import PLAYBACK_SUPPORTED_FORMATS, OpenAITTS
from core.utils.config import ConfigManager
from core.utils.logger import logger


class TTSRouter:
    """Resolve a backend's TTS settings and play the generated response."""

    XIAOAI_TTS_SPEAKER = "xiaoai"
    SUPPORTED_PROVIDERS = frozenset(("xiaoai", "doubao", "openai", "mlx_audio"))

    @classmethod
    def resolve_provider(
        cls,
        configured_provider: str | None,
        tts_speaker: str | None,
    ) -> str:
        """Resolve an explicit provider while preserving legacy selection."""
        provider = str(configured_provider or "").strip().lower()
        if provider in cls.SUPPORTED_PROVIDERS:
            return provider
        if provider:
            logger.warning(
                f"Unknown tts_provider={configured_provider!r}; "
                "using legacy speaker-based selection"
            )
        return "xiaoai" if tts_speaker == cls.XIAOAI_TTS_SPEAKER else "doubao"

    @classmethod
    async def play(
        cls,
        text: str,
        *,
        configured_provider: str | None,
        tts_speaker: str | None,
        tts_speed: float = 1.0,
        playback_token: int | None = None,
        log_prefix: str = "TTS",
    ) -> None:
        """Synthesize and play text, falling back to XiaoAI native TTS."""
        provider = cls.resolve_provider(configured_provider, tts_speaker)
        try:
            if provider == "xiaoai":
                from core.ref import get_speaker

                speaker = get_speaker()
                if speaker:
                    await speaker.play(text=text, blocking=True)
                return

            if provider == "doubao":
                await cls._play_doubao(
                    text,
                    tts_speaker=tts_speaker,
                    tts_speed=tts_speed,
                    playback_token=playback_token,
                )
                return

            await cls._play_openai_compatible(
                text,
                provider=provider,
                tts_speaker=tts_speaker,
            )
        except Exception as exc:
            logger.error(
                f"[{log_prefix}] Error playing response with TTS: "
                f"{type(exc).__name__}: {exc}"
            )
            await cls._fallback_to_xiaoai(text, log_prefix=log_prefix)

    @classmethod
    async def _play_doubao(
        cls,
        text: str,
        *,
        tts_speaker: str | None,
        tts_speed: float,
        playback_token: int | None,
    ) -> None:
        tts_config = ConfigManager.instance().get_app_config("tts.doubao", {})
        app_id = tts_config.get("app_id")
        access_key = tts_config.get("access_key")
        if not app_id or not access_key:
            raise ValueError("Doubao TTS credentials are not configured")

        speaker_id = (
            tts_speaker
            if tts_speaker and tts_speaker != cls.XIAOAI_TTS_SPEAKER
            else tts_config.get("default_speaker", "zh_female_xiaohe_uranus_bigtts")
        )
        tts = DoubaoTTS(
            app_id=app_id,
            access_key=access_key,
            speaker=speaker_id,
        )
        resolved_format = tts.resolve_audio_format(text)
        if tts_config.get("stream", False):
            await open_xiaoai_server.tts_stream_play(
                text,
                app_id=app_id,
                access_key=access_key,
                resource_id=tts.resource_id,
                speaker=speaker_id,
                speed=tts_speed,
                format=resolved_format,
                sample_rate=24000,
                playback_token=playback_token,
            )
        else:
            await open_xiaoai_server.tts_play(
                text,
                app_id=app_id,
                access_key=access_key,
                resource_id=tts.resource_id,
                speaker=speaker_id,
                speed=tts_speed,
                format=resolved_format,
                sample_rate=24000,
                playback_token=playback_token,
            )

    @classmethod
    async def _play_openai_compatible(
        cls,
        text: str,
        *,
        provider: str,
        tts_speaker: str | None,
    ) -> None:
        if provider == "mlx_audio":
            tts_class = MLXAudioTTS
            config_key = "tts.mlx_audio"
            log_name = "MLX-Audio"
            file_prefix = "mlx_audio_tts_"
        else:
            tts_class = OpenAITTS
            config_key = "tts.openai"
            log_name = "OpenAI"
            file_prefix = "openai_tts_"

        tts_config = ConfigManager.instance().get_app_config(config_key, {})
        voice_override = (
            tts_speaker
            if tts_speaker and tts_speaker != cls.XIAOAI_TTS_SPEAKER
            else None
        )
        tts = tts_class.from_config(tts_config, voice_override=voice_override)
        logger.info(
            f"[{log_name} TTS] Requesting speech: "
            f"url={tts.speech_url}, model={tts.model}, voice={tts.voice}, "
            f"format={tts.response_format}"
        )

        if tts.response_format not in PLAYBACK_SUPPORTED_FORMATS:
            raise ValueError(
                f"response_format={tts.response_format!r} is not supported by "
                "the Bridge playback decoder; use mp3, flac, wav, ogg, or pcm"
            )

        temporary_path: str | None = None
        try:
            audio = await tts.synthesize(text)
            safe_format = "".join(
                character
                for character in tts.response_format.lower()
                if character.isalnum()
            ) or "wav"
            with tempfile.NamedTemporaryFile(
                prefix=file_prefix,
                suffix=f".{safe_format}",
                delete=False,
            ) as audio_file:
                temporary_path = audio_file.name
                audio_file.write(audio)

            from core.ref import get_speaker

            speaker = get_speaker()
            if not speaker:
                raise RuntimeError("Speaker manager is not available")
            played = await speaker.play_server_file(
                file_path=temporary_path,
                blocking=True,
            )
            if played is False:
                raise RuntimeError("SpeakerManager.play_server_file returned false")
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    logger.warning(
                        f"[{log_name} TTS] Failed to remove temporary "
                        f"audio file {temporary_path}: {exc}"
                    )

    @classmethod
    async def _fallback_to_xiaoai(cls, text: str, *, log_prefix: str) -> None:
        try:
            from core.ref import get_speaker

            speaker = get_speaker()
            if speaker:
                await speaker.play(text=text, blocking=True)
        except Exception as exc:
            logger.error(f"[{log_prefix}] XiaoAI TTS fallback failed: {exc}")
