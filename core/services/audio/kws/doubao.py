import array
import re
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path

from core.services.audio.asr import ASRManager
from core.utils.config import ConfigManager
from core.utils.logger import logger


class _DoubaoWakeupDetector:
    def __init__(self):
        self.config = ConfigManager.instance()
        self.sample_rate = 16000
        self.sample_width = 2  # int16 mono
        self.buffer = deque()
        self.buffer_bytes = 0
        self.last_check_at = 0.0
        self.last_trigger_at = 0.0
        self.in_utterance = False
        self.utterance_silence_frames = 0
        self.apply_runtime_config()
        self.config.add_reload_listener(self._on_config_reload)

    def _on_config_reload(self, *_args):
        self.apply_runtime_config()

    def apply_runtime_config(self):
        wakeup_cfg = self.config.get_app_config("wakeup", {}) or {}
        doubao_cfg = wakeup_cfg.get("doubao", {}) or {}
        self.keywords = [str(x).strip().lower() for x in wakeup_cfg.get("keywords", []) if str(x).strip()]
        self.mode = str(doubao_cfg.get("mode", "rolling_window") or "rolling_window").strip().lower()
        if self.mode not in {"rolling_window", "utterance_end"}:
            logger.warning(f"未知 wakeup.doubao.mode={self.mode!r}，回退到 rolling_window", module="KWS")
            self.mode = "rolling_window"
        wakeup_asr_provider = doubao_cfg.get("asr_provider")
        if wakeup_asr_provider in (None, "", False):
            contexts = self.config.get_app_config("asr.contexts", {}) or {}
            wakeup_asr_provider = ((contexts.get("wakeup", {}) or {}).get("provider")) if isinstance(contexts, dict) else None
        self.asr_provider = str(wakeup_asr_provider or "openai_compatible").strip().lower()
        if self.asr_provider not in {"sherpa", "openai_compatible"}:
            logger.warning(f"未知 wakeup.doubao.asr_provider={self.asr_provider!r}，回退到 openai_compatible", module="KWS")
            self.asr_provider = "openai_compatible"
        self.window_ms = int(doubao_cfg.get("window_ms", 3500))
        self.check_interval_ms = int(doubao_cfg.get("check_interval_ms", 1800))
        self.min_gap_ms = int(doubao_cfg.get("min_gap_ms", 1500))
        self.min_audio_bytes = int(doubao_cfg.get("min_audio_bytes", 20000))
        self.max_audio_bytes = int(doubao_cfg.get("max_audio_bytes", 0) or 0)
        self.max_audio_ms = int(doubao_cfg.get("max_audio_ms", 2000) or 0)
        self.debug_cfg = self.config.get_app_config("audio_debug", {}) or {}
        self.debug_context_cfg = ((self.debug_cfg.get("contexts", {}) or {}).get("wakeup", {}) or {}) if isinstance(self.debug_cfg, dict) else {}
        self.debug_save_audio = bool(self.debug_cfg.get("enabled", False) and self.debug_context_cfg.get("save_wav", False))
        self.debug_audio_dir = Path(self.debug_context_cfg.get("dir", "/app/core/debug_wakeup_audio"))
        self.debug_max_files = int(self.debug_context_cfg.get("max_files", 10))
        self.utterance_end_silence_frames = int(doubao_cfg.get("utterance_end_silence_frames", 12))
        self.utterance_max_ms = int(doubao_cfg.get("utterance_max_ms", 5000))
        self.max_buffer_bytes = int(self.sample_rate * self.sample_width * self.window_ms / 1000)
        self.utterance_max_buffer_bytes = int(self.sample_rate * self.sample_width * self.utterance_max_ms / 1000)

    def start(self):
        self.reset()
        logger.info(
            (
                f"[KWS] 豆包在线唤醒检测器启动 "
                f"(mode={self.mode}, asr_provider={self.asr_provider}, window={self.window_ms}ms, check_interval={self.check_interval_ms}ms, min_gap={self.min_gap_ms}ms)"
            ),
            module="KWS",
        )

    def reset(self):
        self.buffer.clear()
        self.buffer_bytes = 0
        self.last_check_at = 0.0
        self.in_utterance = False
        self.utterance_silence_frames = 0

    def _append_frames(self, frames: bytes, max_bytes: int | None = None):
        if not frames:
            return
        chunk = bytes(frames)
        self.buffer.append(chunk)
        self.buffer_bytes += len(chunk)
        max_bytes = max_bytes or self.max_buffer_bytes
        while self.buffer_bytes > max_bytes and self.buffer:
            removed = self.buffer.popleft()
            self.buffer_bytes -= len(removed)

    def _snapshot(self) -> bytes:
        if not self.buffer:
            return b""
        return b"".join(self.buffer)

    def _sanitize_label(self, value: str) -> str:
        safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", str(value or ""))
        return safe.strip("._-") or "unknown"

    def _cleanup_debug_audio(self):
        if not self.debug_audio_dir.exists():
            return
        wav_files = sorted(self.debug_audio_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old_file in wav_files[self.debug_max_files:]:
            try:
                old_file.unlink()
            except Exception:
                pass

    def _save_debug_audio(self, pcm_bytes: bytes, status: str, detail: str = ""):
        if not self.debug_save_audio or not pcm_bytes:
            return None
        try:
            self.debug_audio_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
            status_part = self._sanitize_label(status)
            detail_part = self._sanitize_label(detail) if detail else ""
            filename = f"{timestamp}-{status_part}"
            if detail_part:
                filename += f"-{detail_part}"
            filename += f"-{len(pcm_bytes)}b.wav"
            output_path = self.debug_audio_dir / filename
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(self.sample_width)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(pcm_bytes)
            self._cleanup_debug_audio()
            logger.info(f"[KWS] 已保存豆包唤醒调试音频: {output_path}", module="KWS")
            return output_path
        except Exception as exc:
            logger.warning(f"[KWS] 保存豆包唤醒调试音频失败: {exc}", module="KWS")
            return None

    def _recognize(self, pcm_bytes: bytes, now: float):
        if not pcm_bytes:
            return None
        if len(pcm_bytes) < self.min_audio_bytes:
            logger.debug(
                f"[KWS] 豆包唤醒检测跳过：音频过短 ({len(pcm_bytes)} < {self.min_audio_bytes})",
                module="KWS",
            )
            return None
        if self.max_audio_bytes > 0 and len(pcm_bytes) > self.max_audio_bytes:
            logger.info(
                f"[KWS] 豆包唤醒检测丢弃：音频过长 ({len(pcm_bytes)} > {self.max_audio_bytes})",
                module="KWS",
            )
            return None
        duration_ms = len(pcm_bytes) * 1000 / (self.sample_rate * self.sample_width)
        if self.max_audio_ms > 0 and duration_ms > self.max_audio_ms:
            logger.info(
                f"[KWS] 豆包唤醒检测丢弃：音频时长过长 ({duration_ms:.0f}ms > {self.max_audio_ms}ms)",
                module="KWS",
            )
            return None

        pcm_for_asr = pcm_bytes
        try:
            text = (
                ASRManager.asr(
                    pcm_for_asr,
                    sample_rate=self.sample_rate,
                    context="wakeup",
                    provider_override=self.asr_provider,
                )
                or ""
            ).strip()
        except Exception as exc:
            self._save_debug_audio(pcm_for_asr, "failed", type(exc).__name__)
            logger.warning(
                f"Doubao wakeup ASR failed: {type(exc).__name__}: {exc}",
                module="KWS",
            )
            return None

        if not text:
            return None

        normalized = text.lower()
        logger.debug(f"[KWS] 豆包唤醒检测文本: {text}", module="KWS")
        for keyword in self.keywords:
            if keyword and keyword in normalized:
                self._save_debug_audio(pcm_for_asr, "hit", keyword)
                self.last_trigger_at = now
                self.reset()
                logger.info(
                    f"[KWS] 豆包在线唤醒命中: keyword={keyword}, text={text}",
                    module="KWS",
                )
                return keyword
        return None

    def kws(self, frames: bytes):
        self._append_frames(frames, max_bytes=self.max_buffer_bytes)
        now = time.time()

        if (now - self.last_trigger_at) * 1000 < self.min_gap_ms:
            return None

        if self.mode != "rolling_window":
            return None

        if (now - self.last_check_at) * 1000 < self.check_interval_ms:
            return None

        pcm_bytes = self._snapshot()
        if not pcm_bytes:
            return None

        self.last_check_at = now
        return self._recognize(pcm_bytes, now)

    def on_utterance_start(self):
        if self.mode != "utterance_end":
            return
        self.reset()
        self.in_utterance = True

    def on_utterance_frame(self, frames: bytes, is_speech: bool):
        if self.mode != "utterance_end":
            return None

        now = time.time()
        if (now - self.last_trigger_at) * 1000 < self.min_gap_ms:
            return None

        if not self.in_utterance:
            if not is_speech:
                return None
            self.in_utterance = True

        self._append_frames(frames, max_bytes=self.utterance_max_buffer_bytes)

        if is_speech:
            self.utterance_silence_frames = 0
            return None

        self.utterance_silence_frames += 1
        if self.utterance_silence_frames < self.utterance_end_silence_frames:
            return None

        pcm_bytes = self._snapshot()
        result = self._recognize(pcm_bytes, now)
        if not result:
            self.reset()
        return result

    def on_utterance_end(self):
        if self.mode != "utterance_end":
            return None
        pcm_bytes = self._snapshot()
        self.reset()
        if not pcm_bytes:
            return None
        return self._recognize(pcm_bytes, time.time())


DoubaoWakeupDetector = _DoubaoWakeupDetector()
