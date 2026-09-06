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
    """Dispatches wakeup events to XiaoZhi or external backend controllers."""

    def __init__(self):
        self.config = ConfigManager.instance()

        # =====================================================
        # OpenClaw
        # =====================================================

        self._openclaw_controller = None
        self._openclaw_task: asyncio.Task | None = None

        # =====================================================
        # OpenAI
        # =====================================================

        self._openai_controller = None
        self._openai_task: asyncio.Task | None = None

        # =====================================================
        # QwenPaw
        # =====================================================

        self._qwenpaw_controller = None
        self._qwenpaw_task: asyncio.Task | None = None

        # =====================================================
        # Home Assistant
        # =====================================================

        self._homeassistant_controller = None
        self._homeassistant_task: asyncio.Task | None = None

        # =====================================================
        # XiaoZhi
        # =====================================================

        self._xiaozhi_future: asyncio.Future | None = None

    # =========================================================
    # Event loop
    # =========================================================

    def _get_loop(self):
        app = get_app()

        if app:
            return app.loop

        from core.xiaoai import XiaoAI

        return XiaoAI.async_loop

    # =========================================================
    # Device playback
    # =========================================================

    async def _stop_device_playback(self):
        speaker = get_speaker()

        if speaker:
            await speaker.stop_device_audio()

            import open_xiaoai_server

            await open_xiaoai_server.start_recording()

            return

        import open_xiaoai_server

        await open_xiaoai_server.stop_playing()

        await open_xiaoai_server.start_recording()

    # =========================================================
    # Interrupt
    # =========================================================

    def on_interrupt(self):
        logger.info(
            "[Wakeup] XiaoAI wakeup — "
            "interrupting active sessions"
        )

        loop = self._get_loop()

        # =====================================================
        # XiaoZhi
        # =====================================================

        if (
            self._xiaozhi_future
            and not self._xiaozhi_future.done()
        ):
            self._xiaozhi_future.cancel()

        self._xiaozhi_future = None

        xiaozhi = get_xiaozhi()

        if xiaozhi:
            xiaozhi.stop_wakeup_session()

        # =====================================================
        # OpenClaw
        # =====================================================

        if (
            self._openclaw_controller
            and self._openclaw_controller.is_active()
        ):
            self._openclaw_controller.stop()

        if (
            self._openclaw_task
            and not self._openclaw_task.done()
        ):
            loop.call_soon_threadsafe(
                self._openclaw_task.cancel
            )

        # =====================================================
        # OpenAI
        # =====================================================

        if (
            self._openai_controller
            and self._openai_controller.is_active()
        ):
            self._openai_controller.stop()

        if (
            self._openai_task
            and not self._openai_task.done()
        ):
            loop.call_soon_threadsafe(
                self._openai_task.cancel
            )

        # =====================================================
        # QwenPaw
        # =====================================================

        if (
            self._qwenpaw_controller
            and self._qwenpaw_controller.is_active()
        ):
            self._qwenpaw_controller.stop()

        if (
            self._qwenpaw_task
            and not self._qwenpaw_task.done()
        ):
            loop.call_soon_threadsafe(
                self._qwenpaw_task.cancel
            )

        # =====================================================
        # Home Assistant
        # =====================================================

        if (
            self._homeassistant_controller
            and self._homeassistant_controller.is_active()
        ):
            self._homeassistant_controller.stop()

        if (
            self._homeassistant_task
            and not self._homeassistant_task.done()
        ):
            loop.call_soon_threadsafe(
                self._homeassistant_task.cancel
            )

        # =====================================================
        # Stop device playback
        # =====================================================

        asyncio.run_coroutine_threadsafe(
            self._stop_device_playback(),
            loop,
        )

        from core.xiaoai import XiaoAI

        XiaoAI.stop_conversation()

    # =========================================================
    # XiaoZhi wakeup
    # =========================================================

    def on_wakeup(self):
        logger.info(
            "[Wakeup] Wakeup session started"
        )

        xiaozhi = get_xiaozhi()

        if xiaozhi:
            xiaozhi._is_first_round = True

            future = asyncio.run_coroutine_threadsafe(
                xiaozhi.start_wakeup_session(),
                self._get_loop(),
            )

            self._xiaozhi_future = future

            def _clear_future(
                done_future,
            ):
                if (
                    self._xiaozhi_future
                    is done_future
                ):
                    self._xiaozhi_future = None

            future.add_done_callback(
                _clear_future
            )

    # =========================================================
    # Speech
    # =========================================================

    def on_speech(
        self,
        speech_buffer: bytes,
    ):
        pass

    # =========================================================
    # Silence
    # =========================================================

    def on_silence(self):
        pass

    # =========================================================
    # XiaoAI native ASR routing
    # =========================================================

    def consume_xiaoai_asr_result(
        self,
        dialog_id: str,
        text: str,
        is_final,
        is_vad_begin,
    ) -> bool:
        """
        Route XiaoAI native ASR result to the active
        external conversation controller.
        """

        # =====================================================
        # Diagnostic log
        # =====================================================

        logger.info(
            f"[Wakeup] XiaoAI ASR: "
            f"dialog_id={dialog_id}, "
            f"text={text!r}, "
            f"is_final={is_final}, "
            f"is_vad_begin={is_vad_begin}",
            module="Wakeup",
        )

        # =====================================================
        # Home Assistant
        # =====================================================

        if (
            self._homeassistant_controller
            and self._homeassistant_controller.is_active()
        ):
            consumed = (
                self._homeassistant_controller
                .consume_xiaoai_recognize_result(
                    dialog_id=dialog_id,
                    text=text,
                    is_final=is_final,
                    is_vad_begin=is_vad_begin,
                )
            )

            if consumed:
                return True

        # =====================================================
        # OpenClaw
        # =====================================================

        if (
            self._openclaw_controller
            and self._openclaw_controller.is_active()
        ):
            consumed = (
                self._openclaw_controller
                .consume_xiaoai_recognize_result(
                    dialog_id=dialog_id,
                    text=text,
                    is_final=is_final,
                    is_vad_begin=is_vad_begin,
                )
            )

            if consumed:
                return True

        # =====================================================
        # OpenAI
        # =====================================================

        if (
            self._openai_controller
            and self._openai_controller.is_active()
        ):
            consumed = (
                self._openai_controller
                .consume_xiaoai_recognize_result(
                    dialog_id=dialog_id,
                    text=text,
                    is_final=is_final,
                    is_vad_begin=is_vad_begin,
                )
            )

            if consumed:
                return True

        # =====================================================
        # QwenPaw
        # =====================================================

        if (
            self._qwenpaw_controller
            and self._qwenpaw_controller.is_active()
        ):
            consumed = (
                self._qwenpaw_controller
                .consume_xiaoai_recognize_result(
                    dialog_id=dialog_id,
                    text=text,
                    is_final=is_final,
                    is_vad_begin=is_vad_begin,
                )
            )

            if consumed:
                return True

        return False

    # =========================================================
    # Wakeup dispatcher
    # =========================================================

    async def wakeup(
        self,
        text,
        source,
    ):
        before_wakeup = (
            self.config.get_app_config(
                "wakeup.before_wakeup"
            )
        )

        kws = get_kws()

        logger.debug(
            f"[Wakeup] Received wakeup request "
            f"from {source}: {text}"
        )

        # =====================================================
        # Default session keys
        # =====================================================

        from core.openclaw import (
            OpenClawManager,
        )

        default_session_key = (
            self.config.get_app_config(
                "openclaw",
                {},
            ).get(
                "session_key",
                "agent:main:open-xiaoai-bridge",
            )
        )

        OpenClawManager._session_key = (
            default_session_key
        )

        # -----------------------------------------------------

        from core.openai import (
            OpenAIManager,
        )

        default_openai_session_key = (
            self.config.get_app_config(
                "openai",
                {},
            ).get(
                "session_key",
                "agent:default:open-xiaoai-bridge",
            )
        )

        OpenAIManager._session_key = (
            default_openai_session_key
        )

        # -----------------------------------------------------

        from core.qwenpaw import (
            QwenPawManager,
        )

        default_qwenpaw_session_key = (
            self.config.get_app_config(
                "qwenpaw",
                {},
            ).get(
                "session_key",
                "agent:default:open-xiaoai-bridge",
            )
        )

        QwenPawManager._session_key = (
            default_qwenpaw_session_key
        )

        # -----------------------------------------------------
        # Home Assistant session key
        # -----------------------------------------------------

        from core.homeassistant import (
            HomeAssistantManager,
        )

        default_homeassistant_session_key = (
            self.config.get_app_config(
                "homeassistant",
                {},
            ).get(
                "session_key",
                "homeassistant:open-xiaoai-bridge",
            )
        )

        HomeAssistantManager._session_key = (
            default_homeassistant_session_key
        )

        # =====================================================
        # before_wakeup
        # =====================================================

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

        logger.info(
            f"[Wakeup] before_wakeup returned: "
            f"{should_wakeup}"
        )

        if should_wakeup is not None:
            await self.reset_all_sessions()

        # =====================================================
        # Dispatch
        # =====================================================

        if should_wakeup == "openclaw":
            await self._start_openclaw_conversation()

        elif should_wakeup == "openai":
            await self._start_openai_conversation()

        elif should_wakeup == "qwenpaw":
            await self._start_qwenpaw_conversation()

        elif should_wakeup == "homeassistant":
            await self._start_homeassistant_conversation(
                source=source,
                wake_word=text,
            )

        elif should_wakeup == "xiaozhi":
            self.on_wakeup()

    # =========================================================
    # OpenClaw
    # =========================================================

    async def _start_openclaw_conversation(
        self,
    ):
        from core.openclaw_conversation import (
            OpenClawConversationController,
        )

        kws = get_kws()

        if kws:
            kws.pause()

        try:
            self._openclaw_controller = (
                OpenClawConversationController()
            )

            self._openclaw_task = (
                asyncio.create_task(
                    self._openclaw_controller.start()
                )
            )

            await self._openclaw_task

        except asyncio.CancelledError:
            pass

        except Exception as exc:
            logger.error(
                "[Wakeup] OpenClaw conversation failed: "
                f"{type(exc).__name__}: {exc}",
                module="Wakeup",
            )

        finally:
            self._openclaw_controller = None

            self._openclaw_task = None

            if kws:
                kws.resume()

    # =========================================================
    # OpenAI
    # =========================================================

    async def _start_openai_conversation(
        self,
    ):
        from core.openai_conversation import (
            OpenAIConversationController,
        )

        kws = get_kws()

        if kws:
            kws.pause()

        try:
            self._openai_controller = (
                OpenAIConversationController()
            )

            self._openai_task = (
                asyncio.create_task(
                    self._openai_controller.start()
                )
            )

            await self._openai_task

        except asyncio.CancelledError:
            pass

        except Exception as exc:
            logger.error(
                "[Wakeup] OpenAI conversation failed: "
                f"{type(exc).__name__}: {exc}",
                module="Wakeup",
            )

        finally:
            self._openai_controller = None

            self._openai_task = None

            if kws:
                kws.resume()

    # =========================================================
    # QwenPaw
    # =========================================================

    async def _start_qwenpaw_conversation(
        self,
    ):
        from core.qwenpaw_conversation import (
            QwenPawConversationController,
        )

        kws = get_kws()

        if kws:
            kws.pause()

        try:
            self._qwenpaw_controller = (
                QwenPawConversationController()
            )

            self._qwenpaw_task = (
                asyncio.create_task(
                    self._qwenpaw_controller.start()
                )
            )

            await self._qwenpaw_task

        except asyncio.CancelledError:
            pass

        except Exception as exc:
            logger.error(
                "[Wakeup] QwenPaw conversation failed: "
                f"{type(exc).__name__}: {exc}",
                module="Wakeup",
            )

        finally:
            self._qwenpaw_controller = None

            self._qwenpaw_task = None

            if kws:
                kws.resume()

    # =========================================================
    # Home Assistant
    # =========================================================

    async def _start_homeassistant_conversation(
        self,
        source=None,
        wake_word=None,
    ):
        from core.homeassistant_conversation import (
            HomeAssistantConversationController,
        )

        kws = get_kws()

        if kws:
            kws.pause()

        try:
            self._homeassistant_controller = (
                HomeAssistantConversationController(
                    source=source,
                    wake_word=wake_word,
                )
            )

            self._homeassistant_task = (
                asyncio.create_task(
                    self._homeassistant_controller.start()
                )
            )

            await self._homeassistant_task

        except asyncio.CancelledError:
            pass

        except Exception as exc:
            logger.error(
                "[Wakeup] Home Assistant conversation failed: "
                f"{type(exc).__name__}: {exc}",
                module="Wakeup",
            )

        finally:
            self._homeassistant_controller = None

            self._homeassistant_task = None

            if kws:
                kws.resume()

    # =========================================================
    # Reset all sessions
    # =========================================================

    async def reset_all_sessions(self):
        from core.xiaoai import XiaoAI

        from core.ref import get_xiaozhi

        # =====================================================
        # Stop XiaoAI conversation
        # =====================================================

        XiaoAI.stop_conversation()

        # =====================================================
        # Stop XiaoZhi
        # =====================================================

        xiaozhi = get_xiaozhi()

        if (
            xiaozhi
            and xiaozhi.is_connected()
        ):
            try:
                await xiaozhi.send_abort_speaking(
                    AbortReason.ABORT
                )
            except Exception:
                pass

        # =====================================================
        # OpenClaw
        # =====================================================

        if (
            self._openclaw_controller
            and self._openclaw_controller.is_active()
        ):
            self._openclaw_controller.stop()

        # =====================================================
        # OpenAI
        # =====================================================

        if (
            self._openai_controller
            and self._openai_controller.is_active()
        ):
            self._openai_controller.stop()

        # =====================================================
        # QwenPaw
        # =====================================================

        if (
            self._qwenpaw_controller
            and self._qwenpaw_controller.is_active()
        ):
            self._qwenpaw_controller.stop()

        # =====================================================
        # Home Assistant
        # =====================================================

        if (
            self._homeassistant_controller
            and self._homeassistant_controller.is_active()
        ):
            self._homeassistant_controller.stop()

        # =====================================================
        # Stop device playback
        # =====================================================

        await self._stop_device_playback()

        logger.debug(
            "[Wakeup] All sessions reset"
        )

        # =========================================================
# Global event manager
# =========================================================

EventManager = WakeupSessionManager()