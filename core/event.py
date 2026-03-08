import asyncio

from config import APP_CONFIG
from core.utils.logger import logger
from core.ref import (
    get_app,
    get_audio_codec,
    get_kws,
    get_speaker,
    get_vad,
    get_xiaoai,
    get_xiaozhi,
    set_speech_frames,
)
from core.services.protocols.typing import AbortReason, DeviceState, ListeningMode
from core.utils.base import get_env


class Step:
    idle = "idle"
    on_interrupt = "on_interrupt"
    on_wakeup = "on_wakeup"
    on_tts_start = "on_tts_start"
    on_tts_end = "on_tts_end"
    on_speech = "on_speech"
    on_silence = "on_silence"


class __EventManager:
    def __init__(self):
        self.session_id = 0
        self.current_step = Step.idle
        self.next_step_future = None

    def update_step(self, step: Step, step_data=None):
        if not get_env("CLI"):
            return

        self.current_step = step
        if self.next_step_future:
            get_xiaoai().async_loop.call_soon_threadsafe(
                self.next_step_future.set_result, (step, step_data)
            )
            self.next_step_future = None

    async def wait_next_step(self, timeout=None):
        current_session = self.session_id

        self.next_step_future = get_xiaoai().async_loop.create_future()

        async def _timeout(timeout):
            idx = 0
            while idx < timeout:
                idx += 1
                await asyncio.sleep(1)
            return ("timeout", None)

        futures = [self.next_step_future]

        if timeout:
            futures.append(get_xiaoai().async_loop.create_task(_timeout(timeout)))

        done, _ = await asyncio.wait(
            futures,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if current_session != self.session_id:
            # 当前 session 已经结束
            return ("interrupted", None)
        return list(done)[0].result()

    def on_interrupt(self):
        """用户打断（小爱同学）"""
        self.session_id = self.session_id + 1
        self.update_step(Step.on_interrupt)
        self.start_session()

    def on_wakeup(self):
        """用户唤醒（你好小智）"""
        self.session_id = self.session_id + 1
        self.update_step(Step.on_wakeup)
        self.start_session()

    def on_tts_end(self, session_id):
        """TTS结束"""
        if self.current_step not in [Step.on_tts_start]:
            # 当前 session 已经被打断了，不再处理
            return
        self.session_id = self.session_id + 1
        self.update_step(Step.on_tts_end)
        self.start_session()

    def on_tts_start(self, session_id):
        """TTS开始"""
        self.update_step(Step.on_tts_start)

    def on_speech(self, speech_buffer: bytes):
        """检测到声音（开始说话"""
        self.update_step(Step.on_speech, speech_buffer)

    def on_silence(self):
        """检测到静音（说话结束）"""
        self.update_step(Step.on_silence)

    def start_session(self):
        asyncio.run_coroutine_threadsafe(
            self.__start_session(), get_xiaoai().async_loop
        )

    async def __start_session(self):
        if not get_env("CLI"):
            return

        vad = get_vad()
        codec = get_audio_codec()
        speaker = get_speaker()
        xiaozhi = get_xiaozhi()

        # 先取消之前的 VAD 检测和音频输入输出流
        xiaozhi.set_device_state(DeviceState.IDLE)
        
        await xiaozhi.protocol.send_abort_speaking(AbortReason.ABORT)

        # 小爱同学唤醒时，直接打断
        if self.current_step == Step.on_interrupt:
            return

        # 等待 TTS 余音结束
        if self.current_step in [Step.on_tts_end]:
            vad.resume("silence")
            step, _ = await self.wait_next_step()
            if step != Step.on_silence:
                logger.warning(f"{step} != {Step.on_silence} -- tts")
                return

        # 检查是否有人说话
        vad.resume("speech")
        step, speech_buffer = await self.wait_next_step(
            timeout=APP_CONFIG["wakeup"]["timeout"]
        )
        if step == "timeout":
            # 如果没人说话，则回到 IDLE 状态
            xiaozhi.set_device_state(DeviceState.IDLE)
            logger.info("👋 已退出唤醒")
            after_wakeup = APP_CONFIG["wakeup"]["after_wakeup"]
            await after_wakeup(speaker)
            return
        if step != Step.on_speech:
            logger.warning(f"{step} != {Step.on_speech} -- timeout")
            return

        # 开始说话
        logger.debug(f"开始说话...., speech_buffer size: {len(speech_buffer)}")
        set_speech_frames(speech_buffer)
        codec.input_stream.start_stream()  # 开启录音
        await xiaozhi.protocol.send_start_listening(ListeningMode.MANUAL)
        xiaozhi.set_device_state(DeviceState.LISTENING)

        # 等待说话结束
        vad.resume("silence")
        step, _ = await self.wait_next_step()
        if step != Step.on_silence:
            logger.warning(f"{step} != {Step.on_silence} -- silence")
            return

        # 停止说话
        logger.info("---说话结束---")
        await xiaozhi.protocol.send_stop_listening()
        xiaozhi.set_device_state(DeviceState.IDLE)

    async def wakeup(self, text, source):
        before_wakeup = APP_CONFIG["wakeup"]["before_wakeup"]
        get_kws().pause()  # 暂停 KWS 检测
        wakeup = await before_wakeup(get_speaker(), text, source, get_xiaozhi(), get_xiaoai(), get_app())
        get_kws().resume()  # 恢复 KWS 检测
        if wakeup:
            self.on_wakeup()


EventManager = __EventManager()
