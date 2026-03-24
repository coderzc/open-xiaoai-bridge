import asyncio

from core.ref import (
    get_app,
    get_kws,
    get_speaker,
    get_xiaozhi,
)
from core.services.protocols.typing import AbortReason
from core.utils.config import ConfigManager
from core.utils.logger import logger


class WakeupSessionManager:
    """Dispatches wakeup events to XiaoZhi or OpenClaw controllers."""

    def __init__(self):
        self.config = ConfigManager.instance()
        self._openclaw_controller = None
        self._openclaw_task: asyncio.Task | None = None

    def _get_loop(self):
        app = get_app()
        if app:
            return app.loop
        from core.xiaoai import XiaoAI
        return XiaoAI.async_loop

    async def _stop_device_playback(self):
        """Stop all audio playback on the device and restart recording.

        - killall tts_play.sh miplayer: stop blocking TTS (tts_play.sh + child miplayer)
        - mphelper pause: stop non-blocking TTS (mibrain text_to_speech via mediaplayer)
        - stop_playing: kill aplay (our PCM channel)
        - start_playing / start_recording: restart audio streams
        """
        import open_xiaoai_server
        speaker = get_speaker()
        if speaker:
            await speaker.run_shell(
                "killall tts_play.sh miplayer 2>/dev/null; mphelper pause"
            )
        await open_xiaoai_server.stop_playing()
        await open_xiaoai_server.start_playing()
        await open_xiaoai_server.start_recording()

    def on_interrupt(self):
        logger.info("[Wakeup] XiaoAI wakeup — interrupting active sessions")

        loop = self._get_loop()

        # Cancel the OpenClaw asyncio task (interrupts any blocking TTS await)
        if self._openclaw_task and not self._openclaw_task.done():
            loop.call_soon_threadsafe(self._openclaw_task.cancel)
        elif self._openclaw_controller and self._openclaw_controller.is_active():
            self._openclaw_controller.stop()

        asyncio.run_coroutine_threadsafe(self._stop_device_playback(), loop)

        from core.xiaoai import XiaoAI
        XiaoAI.stop_conversation()

    def on_wakeup(self):
        logger.info("[Wakeup] Wakeup session started")
        xiaozhi = get_xiaozhi()
        if xiaozhi:
            xiaozhi._is_first_round = True
            asyncio.run_coroutine_threadsafe(
                xiaozhi.start_wakeup_session(), self._get_loop()
            )

    def on_speech(self, speech_buffer: bytes):
        """Called by VAD when speech is detected."""
        pass

    def on_silence(self):
        """Called by VAD when silence is detected."""
        pass

    async def wakeup(self, text, source):
        before_wakeup = self.config.get_app_config("wakeup.before_wakeup")
        kws = get_kws()
        logger.debug(f"[Wakeup] Received wakeup request from {source}: {text}")

        # Reset session_key to config default before each wakeup,
        # so paths that don't call set_openclaw_session_key() always use the default.
        from core.openclaw import OpenClawManager
        default_session_key = self.config.get_app_config("openclaw", {}).get(
            "session_key", "agent:main:open-xiaoai-bridge"
        )
        OpenClawManager._session_key = default_session_key

        if kws:
            kws.pause()
        should_wakeup = await before_wakeup(
            get_speaker(),
            text,
            source,
            get_app(),
        )
        if kws:
            kws.resume()
        logger.info(f"[Wakeup] before_wakeup returned: {should_wakeup}")
        if should_wakeup is not None:
            await self.reset_all_sessions()

        if should_wakeup == "openclaw":
            await self._start_openclaw_conversation()
        elif should_wakeup == "xiaozhi":
            self.on_wakeup()

    async def _start_openclaw_conversation(self):
        """Start an OpenClaw continuous conversation session.

        This runs independently of the XiaoZhi session state machine.
        KWS is paused during the conversation and resumed when done.
        """
        from core.openclaw_conversation import OpenClawConversationController

        kws = get_kws()
        if kws:
            kws.pause()
        try:
            self._openclaw_controller = OpenClawConversationController()
            self._openclaw_task = asyncio.create_task(self._openclaw_controller.start())
            await self._openclaw_task
        except asyncio.CancelledError:
            pass  # interrupted cleanly by on_interrupt
        except Exception as exc:
            logger.error(
                f"[Wakeup] OpenClaw conversation failed: {type(exc).__name__}: {exc}",
                module="Wakeup",
            )
        finally:
            self._openclaw_controller = None
            self._openclaw_task = None
            if kws:
                kws.resume()

    async def reset_all_sessions(self):
        """Reset all active sessions before starting a new one.

        Stops XiaoAI continuous conversation, interrupts any active XiaoZhi
        session, and stops any OpenClaw continuous conversation.
        """
        from core.xiaoai import XiaoAI
        from core.ref import get_xiaozhi

        # Stop XiaoAI continuous conversation
        XiaoAI.stop_conversation()

        # Interrupt active XiaoZhi session
        xiaozhi = get_xiaozhi()
        if xiaozhi and xiaozhi.is_connected():
            try:
                await xiaozhi.send_abort_speaking(AbortReason.ABORT)
            except Exception:
                pass

        # Stop OpenClaw continuous conversation
        if self._openclaw_controller and self._openclaw_controller.is_active():
            self._openclaw_controller.stop()

        # Stop all audio playback on the device
        await self._stop_device_playback()

        logger.debug("[Wakeup] All sessions reset")


EventManager = WakeupSessionManager()
