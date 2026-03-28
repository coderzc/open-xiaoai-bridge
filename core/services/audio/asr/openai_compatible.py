"""OpenAI-compatible online ASR client.

Used to bridge local PCM audio captured by open-xiaoai-bridge to any
OpenAI-style speech-to-text endpoint such as a local proxy that exposes
`POST /audio/transcriptions` and returns `{"text": "..."}`.
"""

from __future__ import annotations

import io
import wave
from typing import Any

import requests

from core.utils.config import ConfigManager
from core.utils.logger import logger


class _OpenAICompatibleASR:
    def _cfg(self) -> ConfigManager:
        return ConfigManager.instance()

    def _get_base_url(self) -> str:
        return str(
            self._cfg().get_app_config("asr.openai_compatible.base_url", "http://127.0.0.1:8787")
        ).rstrip("/")

    def _get_api_key(self) -> str:
        return str(self._cfg().get_app_config("asr.openai_compatible.api_key", "dummy"))

    def _get_timeout(self) -> float:
        value = self._cfg().get_app_config("asr.timeout", 20)
        try:
            return float(value)
        except Exception:
            return 20.0

    def _get_model(self) -> str:
        return str(self._cfg().get_app_config("asr.model", "bigmodel"))

    def _pcm_to_wav_bytes(self, pcm_bytes: bytes, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buffer.getvalue()

    def _apply_replacements(self, text: str) -> str:
        replacements = self._cfg().get_app_config("asr.replacements", {})
        if not isinstance(replacements, dict):
            return text
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def asr(self, pcm_bytes: bytes, sample_rate: int = 16000) -> str:
        base_url = self._get_base_url()
        timeout = self._get_timeout()
        model = self._get_model()

        wav_bytes = self._pcm_to_wav_bytes(pcm_bytes, sample_rate=sample_rate)
        files = {
            "file": ("speech.wav", wav_bytes, "audio/wav"),
        }
        data: dict[str, Any] = {
            "model": model,
        }
        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
        }

        logger.asr_event(
            "在线识别请求开始",
            f"provider=openai_compatible, model={model}, url={base_url}/audio/transcriptions, wav_bytes={len(wav_bytes)}",
        )

        response = requests.post(
            f"{base_url}/audio/transcriptions",
            headers=headers,
            files=files,
            data=data,
            timeout=timeout,
        )
        response.raise_for_status()

        payload = response.json()
        text = str(payload.get("text", "") or "").strip()
        text = self._apply_replacements(text)

        if text:
            logger.debug(f"[ASR] Online recognized: {text}", module="ASR")
        else:
            logger.debug("[ASR] Online ASR returned empty text", module="ASR")
        return text


OpenAICompatibleASR = _OpenAICompatibleASR()
