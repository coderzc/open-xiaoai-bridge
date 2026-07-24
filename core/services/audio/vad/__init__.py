import threading
import time

from core.ref import set_vad
from core.services.audio.stream import MyAudio
from core.services.audio.vad.silero import Silero
from core.services.protocols.typing import AudioConfig
from core.utils.config import ConfigManager
from core.utils.logger import logger
from core.wakeup_session import EventManager


class _VAD:
    def __init__(self):
        set_vad(self)
        self.config_manager = ConfigManager.instance()

        # 参数设置
        self.sample_rate = 16000
        self.frame_size = 512
        self.threshold = 0.01
        self.min_speech_duration = 250
        self.min_silence_duration = 500

        # 状态变量
        self.paused = True
        self.thread = None
        self.speech_count = 0
        self.silence_count = 0

        self.audio = None
        self.stream = None

        # 暂存的语音片段
        self.silence_frames = []  # 静音片段
        self.speech_frames = []  # 语音片段
        self.target = None  # 检测目标 speech/silence

        self.apply_runtime_config()
        self.config_manager.add_reload_listener(self._on_config_reload)

    def apply_runtime_config(self):
        """同步最新 VAD 配置。"""
        config = self.config_manager.get_app_config("vad", {})
        self.threshold = config.get("threshold", 0.01)
        self.min_speech_duration = config.get("min_speech_duration", 250)
        self.min_silence_duration = config.get("min_silence_duration", 500)

    def _on_config_reload(self, *_args):
        """配置重载后刷新运行时参数。"""
        self.apply_runtime_config()

    def _reset_state(self):
        """重置状态"""
        self.speech_count = 0
        self.silence_count = 0
        self.speech_frames = []
        self.silence_frames = []

    def start(self):
        """启动VAD检测器"""
        logger.vad_event("语音活动检测服务启动", f"最小语音时长={self.min_speech_duration}ms, 最小静音时长={self.min_silence_duration}ms, 阈值={self.threshold}")
        
        self._initialize_audio_stream()

        # 启动检测线程
        self.paused = False
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()

    def pause(self):
        """暂停VAD检测"""
        self.paused = True
        self._reset_state()
        self.stream.stop_stream()

    def resume(self, target: str):
        """恢复VAD检测"""
        self.paused = False
        self.target = target
        self._reset_state()
        self.stream.clear_input()  # discard stale audio + reset read cursor
        self.stream.start_stream()

    def _handle_speech_frame(self, frames):
        """处理语音帧"""
        # speech_count/silence_count are measured in samples (int16 => 2 bytes
        # per sample) so they can be compared against duration_ms * rate / 1000.
        self.speech_count += len(frames) // 2
        self.silence_count = 0

        if self.target == "speech":
            if not self.speech_frames:
                # 加入静音片段（潜在的语音片段）
                self.speech_frames.extend(self.silence_frames)

        # 加入语音片段
        self.speech_frames.extend(frames)

        speech_bytes = bytes(self.speech_frames)

        if (
            self.target == "speech"
            and self.speech_count > self.min_speech_duration * self.sample_rate / 1000
        ):
            self.pause()
            EventManager.on_speech(speech_bytes)

    def _handle_silence_frame(self, frames):
        """处理静音帧"""
        # Count in samples, not bytes (int16 => 2 bytes/sample), to match the
        # min_silence_duration * rate / 1000 threshold below.
        self.silence_count += len(frames) // 2
        self.speech_count = 0

        if self.target == "speech":
            if not self.speech_frames:
                # 如果之前没有语音片段，则将当前帧加入静音片段
                self.silence_frames.extend(frames)
                # 确保静音片段长度不超过 1s
                self.silence_frames = self.silence_frames[
                    -1 * 1 * 2 * self.sample_rate :
                ]
            else:
                # 如果之前有语音片段，则将当前帧加入语音片段
                self.speech_frames.extend(frames)

        if (
            self.target == "silence"
            and self.silence_count > self.min_silence_duration * self.sample_rate / 1000
        ):
            self.pause()
            EventManager.on_silence()

    def _initialize_audio_stream(self):
        """初始化独立的音频流"""
        try:
            # 创建 PyAudio 实例
            self.audio = MyAudio.create()
            # 创建输入流
            self.stream = self.audio.open(
                format=AudioConfig.FORMAT,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size,
                start=True,
            )
            return True
        except Exception:
            return False

    def _close_audio_stream(self):
        """关闭音频流"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            if self.audio:
                self.audio.terminate()
                self.audio = None

        except Exception:
            pass

    def _detection_loop(self):
        """VAD检测主循环"""
        while True:
            # 如果暂停或者音频流未初始化，则跳过
            if self.paused or not self.stream:
                time.sleep(0.1)
                continue

            # 读取缓冲区音频数据
            frames = self.stream.read(self.frame_size)
            if len(frames) != self.frame_size * 2:
                time.sleep(0.01)
                continue

            # 检测是否是语音
            speech_prob = Silero.vad(frames, self.sample_rate) or 0
            is_speech = speech_prob >= self.threshold
            if is_speech:
                self._handle_speech_frame(frames)
            else:
                self._handle_silence_frame(frames)

            time.sleep(0.01)


VAD = _VAD()
