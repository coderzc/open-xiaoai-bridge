import asyncio
from datetime import datetime, timezone

import requests

from core.utils.config import ConfigManager
from core.utils.logger import logger


class HomeAssistantManager:
    """Home Assistant Conversation API manager."""

    CONFIG_PREFIX = "homeassistant"

    def __init__(self):
        self.config = ConfigManager.instance()

        # =====================================================
        # Basic configuration
        # =====================================================

        self.base_url = self._cfg(
            "base_url",
            "",
        ).rstrip("/")

        self.token = self._cfg(
            "token",
            "",
        )

        self.agent_id = self._cfg(
            "agent_id",
            "",
        )

        self.response_timeout = int(
            self._cfg(
                "response_timeout",
                120,
            )
        )

        self.timeout = int(
            self._cfg(
                "timeout",
                30,
            )
        )

        self.keep_conversation = self._cfg(
            "keep_conversation",
            True,
        )

        self.session_key = self._cfg(
            "session_key",
            "homeassistant:open-xiaoai-bridge",
        )

        self.language = self._cfg(
            "language",
            "zh-CN",
        )

        self.exit_keywords = self._cfg(
            "exit_keywords",
            [
                "退出",
                "停止",
                "再见",
                "结束对话",
            ],
        )

        # =====================================================
        # Prompt / TTS
        # =====================================================

        self.rule_prompt = self._cfg(
            "rule_prompt",
            "",
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # ExternalConversationController expects _rule_prompt.
        # Keep this attribute for compatibility.
        #
        # DO NOT append this prompt to the user's text here.
        # Home Assistant Conversation API should receive only
        # the actual user utterance.
        # -----------------------------------------------------

        self._rule_prompt = self.rule_prompt

        self.tts_speed = self._cfg(
            "tts_speed",
            1.0,
        )

        self.tts_speaker = self._cfg(
            "tts_speaker",
            "xiaoai",
        )

        self.intro_prompt = self._cfg(
            "intro_prompt",
            "",
        )

        self.exit_prompt = self._cfg(
            "exit_prompt",
            "好的，再见",
        )

        # =====================================================
        # Conversation state
        # =====================================================

        self.conversation_id = None

        self.turn_count = 0

        self.last_user_text = ""

        self.last_response = ""

        self.last_source = None

        self.last_wake_word = None

        self.continue_conversation = True

        # Keep compatibility with existing architecture.
        self._session_key = self.session_key

    # =========================================================
    # Configuration
    # =========================================================

    def _cfg(
        self,
        key,
        default=None,
    ):
        value = self.config.get_app_config(
            f"{self.CONFIG_PREFIX}.{key}"
        )

        if value is None:
            return default

        return value

    def _state_cfg(
        self,
        key,
        default=None,
    ):
        value = self.config.get_app_config(
            f"{self.CONFIG_PREFIX}.state.{key}"
        )

        if value is None:
            return default

        return value

    # =========================================================
    # Conversation reset
    # =========================================================

    def reset_conversation(self):
        self.conversation_id = None

        self.turn_count = 0

        self.last_user_text = ""

        self.last_response = ""

        self.continue_conversation = True

        logger.debug(
            "[HomeAssistant] Conversation session reset",
            module="HomeAssistant",
        )

    # =========================================================
    # HTTP
    # =========================================================

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _build_payload(
        self,
        text,
    ):
        """
        Build Home Assistant Conversation API payload.

        IMPORTANT:
        Only the actual user utterance is sent as `text`.

        Do NOT do this:

            text = rule_prompt + user_text

        because Home Assistant / Qwen may interpret the
        prompt itself as part of the user's natural-language
        command.
        """

        payload = {
            "text": text,
        }

        if self.language:
            payload["language"] = self.language

        if self.agent_id:
            payload["agent_id"] = self.agent_id

        if (
            self.keep_conversation
            and self.conversation_id
        ):
            payload["conversation_id"] = (
                self.conversation_id
            )

        return payload

    def _post_conversation_sync(
        self,
        payload,
    ):
        url = (
            f"{self.base_url}"
            "/api/conversation/process"
        )

        response = requests.post(
            url,
            headers=self._headers(),
            json=payload,
            timeout=self.response_timeout,
        )

        response.raise_for_status()

        return response.json()

    # =========================================================
    # Send conversation
    # =========================================================

    async def send(
        self,
        text,
        wait_response=True,
    ):
        """
        Send one user utterance to Home Assistant.

        The Home Assistant Conversation Agent is responsible
        for interpreting the command and performing HA tool
        calls.

        Example:

            "打开客厅灯"

        is sent as exactly:

            {"text": "打开客厅灯"}

        NOT:

            {"text": "rule prompt + 打开客厅灯"}
        """

        if not text:
            return ""

        original_text = text.strip()

        if not original_text:
            return ""

        # =====================================================
        # Conversation bookkeeping
        # =====================================================

        self.last_user_text = original_text

        self.turn_count += 1

        payload = self._build_payload(
            original_text
        )

        logger.info(
            "[HomeAssistant] Request: %s",
            original_text,
            module="HomeAssistant",
        )

        logger.debug(
            "[HomeAssistant] Payload: %s",
            payload,
            module="HomeAssistant",
        )

        # =====================================================
        # HTTP request
        # =====================================================

        try:
            result = await asyncio.to_thread(
                self._post_conversation_sync,
                payload,
            )

        except requests.RequestException as exc:
            logger.error(
                "[HomeAssistant] HTTP request failed: "
                f"{type(exc).__name__}: {exc}",
                module="HomeAssistant",
            )
            raise

        except Exception as exc:
            logger.error(
                "[HomeAssistant] Conversation failed: "
                f"{type(exc).__name__}: {exc}",
                module="HomeAssistant",
            )
            raise

        # =====================================================
        # Conversation ID
        # =====================================================

        if self.keep_conversation:
            conversation_id = result.get(
                "conversation_id"
            )

            if conversation_id:
                self.conversation_id = (
                    conversation_id
                )

        else:
            self.conversation_id = None

        # =====================================================
        # Continue conversation
        # =====================================================

        self.continue_conversation = result.get(
            "continue_conversation",
            True,
        )

        # =====================================================
        # Extract speech response
        # =====================================================

        response = result.get(
            "response",
            {},
        )

        speech = response.get(
            "speech",
            {},
        )

        # -----------------------------------------------------
        # Plain speech
        # -----------------------------------------------------

        plain = speech.get(
            "plain",
            {},
        )

        answer = plain.get(
            "speech",
            "",
        )

        # -----------------------------------------------------
        # SSML fallback
        # -----------------------------------------------------

        if not answer:
            ssml = speech.get(
                "ssml",
                {},
            )

            answer = ssml.get(
                "speech",
                "",
            )

        # -----------------------------------------------------
        # Empty response fallback
        # -----------------------------------------------------

        if not answer:
            answer = "好的。"

        self.last_response = answer

        logger.info(
            "[HomeAssistant] Response: %s",
            answer,
            module="HomeAssistant",
        )

        logger.debug(
            "[HomeAssistant] conversation_id=%s, "
            "continue_conversation=%s",
            self.conversation_id,
            self.continue_conversation,
            module="HomeAssistant",
        )

        return answer

    # =========================================================
    # Session context
    # =========================================================

    def set_session_context(
        self,
        source=None,
        wake_word=None,
    ):
        self.last_source = source

        self.last_wake_word = wake_word

    # =========================================================
    # Home Assistant state entity
    # =========================================================

    def _build_state_attributes(self):
        attributes_cfg = self._state_cfg(
            "attributes",
            {},
        )

        all_attributes = {
            "mode": "homeassistant",

            "conversation_id": (
                self.conversation_id
            ),

            "turn_count": self.turn_count,

            "last_user_text": (
                self.last_user_text
            ),

            "last_response": (
                self.last_response
            ),

            "source": self.last_source,

            "wake_word": self.last_wake_word,

            "updated_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

        return {
            key: value
            for key, value in all_attributes.items()
            if attributes_cfg.get(
                key,
                True,
            )
        }

    def _update_state_sync(
        self,
        state,
        attributes,
    ):
        entity_id = self._state_cfg(
            "entity_id"
        )

        if not entity_id:
            return

        url = (
            f"{self.base_url}"
            f"/api/states/{entity_id}"
        )

        response = requests.post(
            url,
            headers=self._headers(),
            json={
                "state": state,
                "attributes": attributes,
            },
            timeout=self.response_timeout,
        )

        response.raise_for_status()

    async def update_state(
        self,
        state=None,
    ):
        """
        Update the Home Assistant state entity.
        """

        if not self._state_cfg(
            "enabled",
            True,
        ):
            return

        if not self._state_cfg(
            "entity_id",
            "",
        ):
            return

        if state is None:
            state = (
                "active"
                if self.continue_conversation
                else "idle"
            )

        attributes = (
            self._build_state_attributes()
        )

        try:
            await asyncio.to_thread(
                self._update_state_sync,
                state,
                attributes,
            )

        except Exception as exc:
            logger.warning(
                "[HomeAssistant] State update failed: "
                f"{type(exc).__name__}: {exc}",
                module="HomeAssistant",
            )

    # =========================================================
    # Start / end session
    # =========================================================

    async def start_session(
        self,
        source=None,
        wake_word=None,
    ):
        """
        Start a new Home Assistant conversation.
        """

        self.reset_conversation()

        self.last_source = source

        self.last_wake_word = wake_word

        await self.update_state(
            "active"
        )

    async def end_session(self):
        """
        End the Home Assistant conversation.
        """

        await self.update_state(
            "idle"
        )

        self.reset_conversation()