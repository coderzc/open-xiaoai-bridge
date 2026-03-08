import asyncio
import os
from re import L
import threading
import time

from config import APP_CONFIG
from core.event import EventManager
from core.ref import get_speaker, get_xiaoai, get_xiaozhi, set_kws
from core.services.audio.kws.sherpa import SherpaOnnx
from core.services.audio.stream import MyAudio
from core.services.audio.vad.silero import Silero
from core.services.protocols.typing import AudioConfig, DeviceState
from core.utils.base import get_env
from core.utils.logger import logger


class _KWS:
    def __init__(self):
        set_kws(self)
        
        # VAD 参数配置
        config = APP_CONFIG.get("vad", {})
        self.vad_threshold = config.get("threshold", 0.10)
        
        # VAD 状态变量
        self.vad_active = False
        self.vad_speech_frames = 0  # 语音帧计数
        self.vad_silence_frames = 0  # 静音帧计数
        self.vad_max_silence_frames = 15  # 15帧（约480ms）静音后停止处理
        self.vad_start_time = time.time()
        
        # 设置帧大小为 512 以兼容 Silero VAD
        self.frame_size = 512
        self.sample_rate = 16000
        self.frame_duration_ms = (self.frame_size * 1000) / self.sample_rate  # 32ms per frame
    def start(self):
        if not get_env("CLI"):
            return

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
        logger.kws_event(f"服务启动 (开始 KWS 检测), 配置：最短静音帧数 {self.vad_max_silence_frames} 帧")

    def get_file_path(self, file_name: str):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "../../../models", file_name)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def _detection_loop(self):
        SherpaOnnx.start()
        self.stream.start_stream()
        while True:
            # 读取缓冲区音频数据
            frames = self.stream.read(self.frame_size)
            if len(frames) != self.frame_size * 2:
                time.sleep(0.01)
                continue

            # 在说话和监听状态时，暂停 KWS
            if (
                not frames
                or self.paused
                or get_xiaozhi().device_state
                in [
                    DeviceState.LISTENING,
                    DeviceState.SPEAKING,
                ]
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
                    logger.debug(f"VAD: 检测到语音，开始 KWS 检测")
                
                self.vad_silence_frames = 0
                
                # 只在有语音时才进行 KWS 检测
                result = SherpaOnnx.kws(frames)
                if result:
                    logger.wakeup(result)
                    self.on_message(result)
                    # 唤醒后重置状态
                    self.vad_active = False
                    self.vad_silence_frames = 0
                    
            else:
                # 静音处理
                if self.vad_active:
                    self.vad_silence_frames += 1
                    
                    # 在激活状态下，允许一定的静音间隙
                    if self.vad_silence_frames <= self.vad_max_silence_frames:
                        # 继续将音频送入 KWS，允许短暂的静音
                        result = SherpaOnnx.kws(frames)
                        if result:
                            logger.wakeup(result)
                            self.on_message(result)
                            self.vad_active = False
                            self.vad_silence_frames = 0
                    else:
                        # 静音超过阈值，停止处理
                        duration_ms = self.vad_silence_frames * self.frame_duration_ms
                        active_duration_ms = (time.time() - self.vad_start_time) * 1000 if hasattr(self, 'vad_start_time') else -1
                        logger.debug(f"VAD: 检测到持续静音（{duration_ms:.0f}ms），暂停 KWS，本次 KWS 监听时长 {active_duration_ms:.0f}ms")
                        self.vad_active = False
                        self.vad_silence_frames = 0

    def on_message(self, text: str):
        asyncio.run_coroutine_threadsafe(
            EventManager.wakeup(text, "kws"),
            get_xiaoai().async_loop,
        )


KWS = _KWS()