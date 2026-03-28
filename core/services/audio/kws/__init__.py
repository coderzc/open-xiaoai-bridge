import asyncio
import os
import threading
import time

from core.ref import get_app, get_xiaoai, get_xiaozhi, set_kws
from core.services.audio.kws.doubao import DoubaoWakeupDetector
from core.services.audio.kws.sherpa import SherpaOnnx
from core.services.audio.stream import MyAudio
from core.services.audio.vad.silero import Silero
from core.services.protocols.typing import AudioConfig, DeviceState
from core.utils.config import ConfigManager
from core.utils.logger import logger
from core.wakeup_session import EventManager


class _KWS:
    def __init__(self):
        set_kws(self)
        self.config_manager = ConfigManager.instance()
        self.vad_threshold = 0.10
        
        # VAD 状态变量
        self.vad_active = False
        self.vad_speech_frames = 0  # 语音帧计数
        self.vad_silence_frames = 0  # 静音帧计数
        self.vad_min_silence_frames = 15  # 静默超过该帧数则判定为说完
        self.vad_start_time = time.time()
        
        # 设置帧大小为 512 以兼容 Silero VAD
        self.frame_size = 512
        self.sample_rate = 16000
        self.frame_duration_ms = (self.frame_size * 1000) / self.sample_rate  # 32ms per frame

        self.apply_runtime_config()
        self.config_manager.add_reload_listener(self._on_config_reload)

    def apply_runtime_config(self):
        """同步最新 KWS 相关配置。"""
        vad_config = self.config_manager.get_app_config("vad", {})
        self.vad_threshold = vad_config.get("threshold", 0.10)
        kws_config = self.config_manager.get_app_config("kws", {})
        min_silence_ms = kws_config.get("min_silence_duration", 480)
        self.vad_min_silence_frames = int(min_silence_ms / self.frame_duration_ms)
        wakeup_cfg = self.config_manager.get_app_config("wakeup", {}) or {}
        provider = wakeup_cfg.get("provider", "local")
        self.wakeup_provider = str(provider or "local").strip().lower()
        if self.wakeup_provider not in {"local", "doubao"}:
            logger.warning(
                f"未知 wakeup.provider={provider!r}，回退到 local",
                module="KWS",
            )
            self.wakeup_provider = "local"

    def _on_config_reload(self, *_args):
        """配置重载后刷新运行时参数。"""
        self.apply_runtime_config()

    def start(self):
        self.audio = MyAudio.create()
        self.stream = self.audio.open(
            format=AudioConfig.FORMAT,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=AudioConfig.FRAME_SIZE,
            start=True,
        )

        # 启动 KWS 服务
        self.paused = False
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()
        config = ConfigManager.instance()
        keywords_score = config.get_app_config("kws.keywords_score", 2.0)
        keywords_threshold = config.get_app_config("kws.keywords_threshold", 0.2)
        min_silence_ms = config.get_app_config("kws.min_silence_duration", 480)
        if self.wakeup_provider == "doubao":
            doubao_cfg = config.get_app_config("wakeup.doubao", {}) or {}
            logger.kws_event(
                "关键词唤醒服务启动",
                (
                    f"provider:doubao, VAD:[threshold:{self.vad_threshold}, min_silence:{min_silence_ms}ms], "
                    f"Doubao:[mode:{doubao_cfg.get('mode', 'rolling_window')}, asr_provider:{doubao_cfg.get('asr_provider', 'openai_compatible')}, window:{doubao_cfg.get('window_ms', 3500)}ms, check_interval:{doubao_cfg.get('check_interval_ms', 1800)}ms, min_gap:{doubao_cfg.get('min_gap_ms', 1500)}ms, min_audio_bytes:{doubao_cfg.get('min_audio_bytes', 20000)}, max_audio_ms:{doubao_cfg.get('max_audio_ms', 2000)}ms, max_audio_bytes:{doubao_cfg.get('max_audio_bytes', 64000)}]"
                ),
            )
        else:
            logger.kws_event("关键词唤醒服务启动", f"provider:local, 关键词:[score:{keywords_score}, threshold:{keywords_threshold}], VAD:[threshold:{self.vad_threshold}, min_silence:{min_silence_ms}ms]")

    def get_file_path(self, file_name: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "../../../models", file_name)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def _detection_loop(self):
        detector = SherpaOnnx if self.wakeup_provider == "local" else DoubaoWakeupDetector
        detector.start()
        self.stream.start_stream()
        while True:
            # 读取缓冲区音频数据
            frames = self.stream.read(self.frame_size)
            if len(frames) != self.frame_size * 2:
                time.sleep(0.01)
                continue

            # 在说话和监听状态时，暂停 KWS
            xiaozhi = get_xiaozhi()
            if (
                not frames
                or self.paused
                or (
                    xiaozhi and xiaozhi.device_state
                    in [DeviceState.LISTENING, DeviceState.SPEAKING]
                )
            ):
                time.sleep(0.01)
                continue

            # 先进行 VAD 检测
            speech_prob = Silero.vad(frames, self.sample_rate) or 0
            is_speech = speech_prob >= self.vad_threshold
            
            if is_speech:
                # 检测到语音，立即激活
                if not self.vad_active:
                    self.vad_active = True
                    self.vad_start_time = time.time()
                    logger.debug("检测到语音，开始 KWS 检测", module="KWS")
                    if hasattr(detector, "on_utterance_start"):
                        detector.on_utterance_start()
                
                self.vad_silence_frames = 0
                
                # 只在有语音时才进行 KWS 检测
                if hasattr(detector, "on_utterance_frame"):
                    result = detector.on_utterance_frame(frames, True)
                else:
                    result = detector.kws(frames)
                if result:
                    logger.wakeup(result, module="KWS")
                    self.on_message(result)
                    # 唤醒后重置状态
                    self.vad_active = False
                    self.vad_silence_frames = 0
                    
            else:
                # 静音处理
                if self.vad_active:
                    self.vad_silence_frames += 1
                    
                    # 在激活状态下，允许一定的静音间隙
                    if self.vad_silence_frames <= self.vad_min_silence_frames:
                        # 继续将音频送入 KWS，允许短暂的静音
                        if hasattr(detector, "on_utterance_frame"):
                            result = detector.on_utterance_frame(frames, False)
                        else:
                            result = detector.kws(frames)
                        if result:
                            logger.wakeup(result, module="KWS")
                            self.on_message(result)
                            self.vad_active = False
                            self.vad_silence_frames = 0
                    else:
                        # 静音超过阈值，停止处理
                        duration_ms = self.vad_silence_frames * self.frame_duration_ms
                        active_duration_ms = (time.time() - self.vad_start_time) * 1000 if hasattr(self, 'vad_start_time') else -1
                        logger.debug(
                            (
                                f"检测到持续静音（{duration_ms:.0f}ms），暂停 KWS，"
                                f"本次 KWS 监听时长 {active_duration_ms:.0f}ms"
                            ),
                            module="KWS",
                        )
                        final_result = None
                        if hasattr(detector, "on_utterance_end"):
                            final_result = detector.on_utterance_end()
                        self.vad_active = False
                        self.vad_silence_frames = 0
                        if final_result:
                            logger.wakeup(final_result, module="KWS")
                            self.on_message(final_result)
                        # Reset detector state to discard partial recognition state,
                        # preventing leftover audio from the previous utterance
                        # from combining with the next one.
                        if hasattr(detector, "reset"):
                            detector.reset()

    def on_message(self, text: str):
        loop = get_app().loop if get_app() else get_xiaoai().async_loop
        logger.debug(f"[KWS] Dispatch wakeup event: {text}")
        future = asyncio.run_coroutine_threadsafe(
            EventManager.wakeup(text, "kws"),
            loop,
        )

        def _log_result(done_future):
            try:
                done_future.result()
            except Exception as exc:
                logger.error(f"[KWS] Wakeup dispatch failed: {type(exc).__name__}: {exc}")

        future.add_done_callback(_log_result)


KWS = _KWS()
