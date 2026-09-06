import asyncio

from core.external_conversation import (
    ExternalConversationController,
)
from core.homeassistant import (
    HomeAssistantManager,
)
from core.ref import get_speaker
from core.utils.config import ConfigManager
from core.utils.logger import logger


class HomeAssistantConversationController(
    ExternalConversationController
):
    """Home Assistant continuous conversation controller."""

    CONFIG_PREFIX = "homeassistant"

    BACKEND_NAME = "Home Assistant"

    LOG_MODULE = "HomeAssistant Conv"

    WAKEUP_SOURCE = "homeassistant"

    # ExternalConversationController 直接使用
    # self.MANAGER，而不是 self.MANAGER()
    #
    # 因此这里必须是实例。
    MANAGER = HomeAssistantManager()

    def __init__(
        self,
        source=None,
        wake_word=None,
    ):
        super().__init__()

        self.source = source

        self.wake_word = wake_word

        self.config = ConfigManager.instance()

        self.backend.set_session_context(
            source=source,
            wake_word=wake_word,
        )

    # =========================================================
    # Start
    # =========================================================

    async def start(self):
        """Start Home Assistant continuous conversation."""

        if not self.backend._cfg(
            "enabled",
            True,
        ):
            logger.info(
                "[HomeAssistant Conv] Disabled by config",
                module=self.LOG_MODULE,
            )
            return

        await self.backend.start_session(
            source=self.source,
            wake_word=self.wake_word,
        )

        intro_prompt = self.backend._cfg(
            "intro_prompt",
            "",
        )

        if intro_prompt:
            speaker = get_speaker()

            if speaker:
                try:
                    await speaker.play(
                        text=intro_prompt
                    )

                except asyncio.CancelledError:
                    raise

                except Exception as exc:
                    logger.warning(
                        "[HomeAssistant Conv] "
                        "Intro TTS failed: "
                        f"{type(exc).__name__}: {exc}",
                        module=self.LOG_MODULE,
                    )

        try:
            await super().start()

        finally:
            await self.backend.end_session()

    # =========================================================
    # TTS
    # =========================================================

    async def _play_tts(
        self,
        response,
    ):
        """
        Home Assistant response -> XiaoAI TTS.

        Do not use OpenClaw/OpenAI/QwenPaw TTS.
        """

        if not response:
            return

        speaker = get_speaker()

        if not speaker:
            logger.warning(
                "[HomeAssistant Conv] "
                "Speaker unavailable",
                module=self.LOG_MODULE,
            )
            return

        try:
            await speaker.play(
                text=str(response)
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            logger.error(
                "[HomeAssistant Conv] "
                "XiaoAI TTS failed: "
                f"{type(exc).__name__}: {exc}",
                module=self.LOG_MODULE,
            )

    # =========================================================
    # XiaoAI Native ASR
    # =========================================================

    async def _run_one_turn_with_xiaoai_asr(
        self,
    ) -> str:
        """
        Execute one Home Assistant conversation turn
        using XiaoAI native ASR.

        Returns:
            continue
            exit
            timeout
            error
        """

        # =====================================================
        # 1. Wait for XiaoAI native ASR
        # =====================================================

        text = (
            await self._wait_for_xiaoai_asr_text()
        )

        if text is None:
            logger.info(
                "[HomeAssistant Conv] "
                "XiaoAI native ASR timeout",
                module=self.LOG_MODULE,
            )

            return "timeout"

        if text == self.XIAOAI_ASR_TIMEOUT:
            logger.debug(
                "[HomeAssistant Conv] "
                "XiaoAI native ASR returned "
                "native timeout",
                module=self.LOG_MODULE,
            )

            return "continue"

        text = text.strip()

        if not text:
            return "continue"

        logger.info(
            f"[HomeAssistant Conv] "
            f"User: {text!r}",
            module=self.LOG_MODULE,
        )

        # =====================================================
        # 2. Exit keywords
        # =====================================================

        normalized = text

        for keyword in self.exit_keywords:
            if (
                keyword
                and keyword in normalized
            ):
                logger.info(
                    f"[HomeAssistant Conv] "
                    f"Exit keyword: {keyword}",
                    module=self.LOG_MODULE,
                )

                return "exit"

        # =====================================================
        # 3. Home Assistant Conversation API
        #
        # DO NOT add _rule_prompt here.
        #
        # HomeAssistantManager.send() already adds it.
        # =====================================================

        try:
            response = await self.backend.send(
                text,
                wait_response=True,
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            logger.error(
                "[HomeAssistant Conv] "
                "Backend failed: "
                f"{type(exc).__name__}: {exc}",
                module=self.LOG_MODULE,
            )

            response = (
                "抱歉，Home Assistant 暂时无法处理这个请求。"
            )

        # =====================================================
        # 4. Update HA state
        # =====================================================

        await self.backend.update_state(
            "active"
        )

        # =====================================================
        # 5. Stop microphone
        # =====================================================

        await self._stop_recording()

        # =====================================================
        # 6. XiaoAI TTS
        # =====================================================

        await self._play_tts(
            response
        )

        # =====================================================
        # 7. Notification sound
        # =====================================================

        await self._play_notify()

        # =====================================================
        # 8. Restart microphone
        # =====================================================

        await self._start_recording()

        logger.debug(
            "[HomeAssistant Conv] "
            "Ready for next XiaoAI native ASR round",
            module=self.LOG_MODULE,
        )

        # =====================================================
        # 9. Check continue_conversation
        # =====================================================

        if not self.backend.continue_conversation:
            logger.info(
                "[HomeAssistant Conv] "
                "Agent requested conversation end",
                module=self.LOG_MODULE,
            )

            return "exit"

        return "continue"