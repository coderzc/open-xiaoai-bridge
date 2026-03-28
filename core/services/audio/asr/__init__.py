import array
import re
import wave
from datetime import datetime
from pathlib import Path

from core.services.audio.asr.openai_compatible import OpenAICompatibleASR
from core.utils.config import ConfigManager
from core.utils.logger import logger


class _ASRManager:
    def _debug_cfg(self) -> dict:
        debug_cfg = ConfigManager.instance().get_app_config("audio_debug", {}) or {}
        return debug_cfg if isinstance(debug_cfg, dict) else {}

    def _context_debug_cfg(self, context: str) -> dict:
        debug_cfg = self._debug_cfg()
        contexts = debug_cfg.get("contexts", {}) or {}
        if isinstance(contexts, dict):
            context_cfg = contexts.get(context, {}) or {}
            if isinstance(context_cfg, dict):
                return context_cfg
        return {}

    def _apply_input_gain(self, pcm_bytes: bytes, gain: float) -> bytes:
        if not pcm_bytes or abs(gain - 1.0) < 1e-6:
            return pcm_bytes
        samples = array.array("h")
        samples.frombytes(pcm_bytes)
        for i, sample in enumerate(samples):
            amplified = int(round(sample * gain))
            if amplified > 32767:
                amplified = 32767
            elif amplified < -32768:
                amplified = -32768
            samples[i] = amplified
        return samples.tobytes()

    def _sanitize_label(self, value: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(value or ""))
        return safe.strip("._-") or "unknown"

    def _cleanup_debug_audio(self, debug_dir: Path, max_files: int):
        if not debug_dir.exists():
            return
        wav_files = sorted(debug_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old_file in wav_files[max_files:]:
            try:
                old_file.unlink()
            except Exception:
                pass

    def _save_debug_audio(self, pcm_bytes: bytes, context: str, status: str, detail: str = ""):
        debug_cfg = self._debug_cfg()
        if not debug_cfg.get("enabled", False) or not pcm_bytes:
            return None
        context_cfg = self._context_debug_cfg(context)
        if not context_cfg.get("save_wav", False):
            return None
        debug_dir = Path(context_cfg.get("dir", f"/tmp/{context}_debug_audio"))
        max_files = int(context_cfg.get("max_files", 10) or 10)
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            filename = f"{timestamp}-{self._sanitize_label(context)}-{self._sanitize_label(status)}"
            if detail:
                filename += f"-{self._sanitize_label(detail)}"
            filename += f"-{len(pcm_bytes)}b.wav"
            output_path = debug_dir / filename
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(pcm_bytes)
            self._cleanup_debug_audio(debug_dir, max_files)
            logger.info(f"[ASR] 已保存调试音频: {output_path}", module="ASR")
            return output_path
        except Exception as exc:
            logger.warning(f"[ASR] 保存调试音频失败: {exc}", module="ASR")
            return None

    def _provider(self, context: str = "conversation") -> str:
        config = ConfigManager.instance()
        contexts = config.get_app_config("asr.contexts", {}) or {}
        if isinstance(contexts, dict):
            context_cfg = contexts.get(context, {}) or {}
            provider = context_cfg.get("provider")
            if provider not in (None, "", False):
                return str(provider)
        provider = config.get_app_config("asr.provider", "sherpa")
        return str(provider or "sherpa")

    def _fallback_provider(self, context: str = "conversation") -> str | None:
        config = ConfigManager.instance()
        contexts = config.get_app_config("asr.contexts", {}) or {}
        if isinstance(contexts, dict):
            context_cfg = contexts.get(context, {}) or {}
            provider = context_cfg.get("fallback_provider")
            if provider not in (None, "", False):
                return str(provider)
        provider = config.get_app_config("asr.fallback_provider", None)
        if provider in (None, "", False):
            return None
        return str(provider)

    def _get_sherpa(self):
        from core.services.audio.asr.sherpa import SherpaASR

        return SherpaASR

    def _run_provider(self, provider: str, pcm_bytes: bytes, sample_rate: int) -> str:
        provider = str(provider or "sherpa")
        if provider == "openai_compatible":
            return OpenAICompatibleASR.asr(pcm_bytes, sample_rate=sample_rate)
        if provider == "sherpa":
            return self._get_sherpa().asr(pcm_bytes, sample_rate=sample_rate)
        raise ValueError(f"Unknown ASR provider: {provider}")

    def asr(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 16000,
        context: str = "conversation",
        provider_override: str | None = None,
        fallback_provider_override: str | None = None,
    ) -> str:
        provider = str(provider_override or self._provider(context=context))
        fallback_provider = (
            str(fallback_provider_override)
            if fallback_provider_override not in (None, "", False)
            else self._fallback_provider(context=context)
        )
        context_debug_cfg = self._context_debug_cfg(context)
        gain = float(context_debug_cfg.get("input_gain", 1.0) or 1.0)
        pcm_for_asr = self._apply_input_gain(pcm_bytes, gain)

        try:
            text = self._run_provider(provider, pcm_for_asr, sample_rate)
            if text:
                self._save_debug_audio(pcm_for_asr, context, "success", provider)
            return text
        except Exception as err:
            self._save_debug_audio(pcm_for_asr, context, "failed", type(err).__name__)
            logger.warning(
                f"ASR provider '{provider}' failed (context={context}): {err}",
                module="ASR",
            )
            if fallback_provider and fallback_provider != provider:
                logger.asr_event(
                    "在线识别失败，回退本地识别",
                    f"context={context}, from={provider}, to={fallback_provider}",
                )
                return self._run_provider(fallback_provider, pcm_for_asr, sample_rate)
            raise

    def warmup(self):
        provider = self._provider(context="conversation")
        if provider == "sherpa":
            self._get_sherpa()._ensure_loaded()
            return
        logger.info(
            f"[ASR] Warmup skipped for provider={provider}",
            module="ASR",
        )


ASRManager = _ASRManager()

__all__ = ["ASRManager", "OpenAICompatibleASR"]
