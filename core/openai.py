"""OpenAI-compatible service manager.

This module talks to any OpenAI-compatible Chat Completions endpoint,
including Hermes Agent API server mode.
"""

import asyncio
import uuid
from typing import Any

import aiohttp

import open_xiaoai_server

from core.utils.base import get_env
from core.utils.config import ConfigManager
from core.utils.logger import logger


class OpenAIManager:
    """Manager for OpenAI-compatible chat backends."""

    XIAOAI_TTS_SPEAKER = "xiaoai"

    _initialized = False
    _reload_listener_registered = False
    _enabled = False
    _base_url = "http://127.0.0.1:8000/v1"
    _api_key = ""
    _model = "gpt-4o-mini"
    _session_key = "default"
    _system_prompt = ""
    _temperature: float | None = None
    _max_tokens: int | None = None
    _timeout = 120
    _history_max_messages = 20
    _extra_body: dict[str, Any] = {}
    _tts_speaker = None
    _session_tts_speakers: dict[str, str] = {}
    _tts_speed = 1.0
    _rule_prompt = ""
    _rule_prompt_for_skill = ""
    _sessions: dict[str, list[dict[str, str]]] = {}
    _response_events: dict[str, asyncio.Future] = {}
    _response_texts: dict[str, str] = {}
    _response_tts_speakers: dict[str, str | None] = {}
    last_error: str | None = None

    @classmethod
    def initialize_from_config(cls, enabled: bool | None = None):
        logger.info("[OpenAI] Initializing from config...")
        cls.reload_from_config(enabled=enabled)
        cls._initialized = True

    @classmethod
    def reload_from_config(cls, enabled: bool | None = None):
        """Refresh OpenAI settings from config.py."""
        config_manager = ConfigManager.instance()
        if not cls._reload_listener_registered:
            config_manager.add_reload_listener(
                lambda _old, _new: cls.reload_from_config()
            )
            cls._reload_listener_registered = True

        config = config_manager.get_app_config("openai", {})

        if enabled is not None:
            cls._enabled = enabled
        else:
            env_enabled = get_env("OPENAI_ENABLE")
            cls._enabled = (
                env_enabled.lower() in ("1", "true", "yes")
                if env_enabled is not None
                else False
            )

        cls._base_url = str(config.get("base_url", "http://127.0.0.1:8000/v1")).rstrip("/")
        cls._api_key = str(config.get("api_key", "") or "")
        cls._model = str(config.get("model", "gpt-4o-mini"))
        cls._session_key = str(config.get("session_key", "default"))
        cls._system_prompt = str(config.get("system_prompt", "") or "")
        cls._timeout = int(config.get("response_timeout", 120))
        cls._history_max_messages = max(0, int(config.get("history_max_messages", 20)))
        cls._temperature = cls._optional_float(config.get("temperature"))
        cls._max_tokens = cls._optional_int(config.get("max_tokens"))
        cls._extra_body = config.get("extra_body", {})
        if not isinstance(cls._extra_body, dict):
            cls._extra_body = {}
        cls._tts_speaker = config.get("tts_speaker", None)
        cls._session_tts_speakers = (
            {
                str(key): str(value)
                for key, value in config.get("session_tts_speakers", {}).items()
                if key and value
            }
            if isinstance(config.get("session_tts_speakers", {}), dict)
            else {}
        )
        cls._tts_speed = float(config.get("tts_speed", 1.0))
        cls._rule_prompt = str(config.get("rule_prompt", "") or "")
        cls._rule_prompt_for_skill = str(config.get("rule_prompt_for_skill", "") or "")

        if cls._enabled:
            logger.info(
                f"[OpenAI] Enabled, base_url={cls._base_url}, model={cls._model}"
            )

    @classmethod
    def _optional_float(cls, value) -> float | None:
        if value is None or value == "":
            return None
        return float(value)

    @classmethod
    def _optional_int(cls, value) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @classmethod
    async def connect(cls) -> bool:
        """No-op connect hook for parity with WebSocket backends."""
        if not cls._initialized:
            cls.initialize_from_config()
        return cls._enabled

    @classmethod
    async def close(cls):
        """Cancel pending response waiters."""
        for waiter in list(cls._response_events.values()):
            if not waiter.done():
                waiter.cancel()
        cls._response_events.clear()
        cls._response_texts.clear()
        cls._response_tts_speakers.clear()

    @classmethod
    def is_connected(cls) -> bool:
        return cls.is_enabled()

    @classmethod
    def is_enabled(cls) -> bool:
        if not cls._initialized:
            cls.initialize_from_config()
        return cls._enabled

    @classmethod
    def set_session_key(cls, session_key: str):
        logger.info(
            f"[OpenAI] Session key updated: {cls._session_key!r} -> {session_key!r}"
        )
        cls._session_key = session_key

    @classmethod
    def reset_session(cls, session_key: str | None = None):
        cls._sessions.pop(session_key or cls._session_key, None)

    @classmethod
    def get_tts_speaker_for_session_key(cls, session_key: str | None = None) -> str | None:
        target_session_key = session_key or cls._session_key
        return cls._session_tts_speakers.get(target_session_key) or cls._tts_speaker

    @classmethod
    async def send(cls, text: str, wait_response: bool = False) -> str | None:
        run_id = await cls._send_and_track(text)
        if run_id is None:
            return None
        if not wait_response:
            asyncio.create_task(cls._wait_response(run_id))
            return run_id
        try:
            return await cls._wait_response(run_id)
        finally:
            cls._response_tts_speakers.pop(run_id, None)

    @classmethod
    async def send_and_play_reply(cls, text: str, wait_response: bool = False) -> str | None:
        run_id = await cls._send_and_track(text)
        if run_id is None:
            return None
        if not wait_response:
            asyncio.create_task(cls._wait_and_play_response(run_id))
            return run_id
        try:
            response_text = await cls._wait_response(run_id)
            if response_text:
                await cls._play_response_with_tts(
                    response_text,
                    tts_speaker=cls._response_tts_speakers.get(run_id),
                )
            return response_text
        finally:
            cls._response_tts_speakers.pop(run_id, None)

    @classmethod
    async def _send_and_track(cls, text: str) -> str | None:
        if not cls._initialized:
            cls.initialize_from_config()
        if not cls._enabled:
            logger.warning("[OpenAI] send called but backend is disabled")
            return None

        run_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        cls._response_events[run_id] = loop.create_future()
        cls._response_texts[run_id] = ""
        cls._response_tts_speakers[run_id] = cls.get_tts_speaker_for_session_key()
        logger.user_speech(text, module=f"OpenAI({cls._session_key})")
        asyncio.create_task(cls._run_chat_completion(run_id, text))
        return run_id

    @classmethod
    async def _wait_response(cls, run_id: str) -> str | None:
        event = cls._response_events.get(run_id)
        if not event:
            logger.warning(f"[OpenAI] No event found for run {run_id}")
            return None
        try:
            await asyncio.wait_for(event, timeout=cls._timeout)
            return cls._response_texts.pop(run_id, "") or None
        except asyncio.TimeoutError:
            logger.warning(f"[OpenAI] Timeout waiting for response (runId: {run_id})")
            return None
        finally:
            cls._response_events.pop(run_id, None)
            cls._response_texts.pop(run_id, None)

    @classmethod
    async def _run_chat_completion(cls, run_id: str, text: str):
        try:
            response_text = await cls._request_chat_completion(text)
            if response_text:
                cls._response_texts[run_id] = response_text
                logger.ai_response(response_text, module=f"OpenAI({cls._session_key})")
        except Exception as exc:
            cls.last_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"[OpenAI] Chat completion failed: {cls.last_error}")
        finally:
            waiter = cls._response_events.get(run_id)
            if waiter and not waiter.done():
                waiter.get_loop().call_soon_threadsafe(waiter.set_result, None)

    @classmethod
    async def _request_chat_completion(cls, text: str) -> str | None:
        session_key = cls._session_key
        history = cls._sessions.setdefault(session_key, [])
        messages = cls._build_messages(history, text)
        payload: dict[str, Any] = {
            "model": cls._model,
            "messages": messages,
            "stream": False,
            **cls._extra_body,
        }
        if cls._temperature is not None:
            payload["temperature"] = cls._temperature
        if cls._max_tokens is not None:
            payload["max_tokens"] = cls._max_tokens

        headers = {"Content-Type": "application/json"}
        if cls._api_key:
            headers["Authorization"] = f"Bearer {cls._api_key}"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=cls._timeout)
        ) as session:
            async with session.post(
                cls._chat_completions_url(),
                json=payload,
                headers=headers,
            ) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    message = body.get("error", body) if isinstance(body, dict) else body
                    raise RuntimeError(f"HTTP {response.status}: {message}")

        response_text = cls._extract_response_text(body)
        if response_text:
            cls._append_history(history, text, response_text)
        return response_text

    @classmethod
    def _chat_completions_url(cls) -> str:
        if cls._base_url.endswith("/chat/completions"):
            return cls._base_url
        return cls._base_url.rstrip("/") + "/chat/completions"

    @classmethod
    def _build_messages(cls, history: list[dict[str, str]], text: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if cls._system_prompt:
            messages.append({"role": "system", "content": cls._system_prompt})
        messages.extend(history[-cls._history_max_messages :] if cls._history_max_messages else [])
        messages.append({"role": "user", "content": text})
        return messages

    @classmethod
    def _append_history(cls, history: list[dict[str, str]], text: str, response_text: str):
        if cls._history_max_messages <= 0:
            return
        history.extend(
            [
                {"role": "user", "content": text},
                {"role": "assistant", "content": response_text},
            ]
        )
        if len(history) > cls._history_max_messages:
            del history[: len(history) - cls._history_max_messages]

    @classmethod
    def _extract_response_text(cls, body: Any) -> str | None:
        if not isinstance(body, dict):
            return None
        choices = body.get("choices")
        if not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                return "".join(parts).strip() or None
        text = first.get("text")
        return text.strip() if isinstance(text, str) else None

    @classmethod
    async def _wait_and_play_response(cls, run_id: str):
        try:
            response_text = await cls._wait_response(run_id)
            if response_text:
                await cls._play_response_with_tts(
                    response_text,
                    tts_speaker=cls._response_tts_speakers.get(run_id),
                )
            else:
                logger.warning(f"[OpenAI] No response text received for run {run_id}")
        finally:
            cls._response_tts_speakers.pop(run_id, None)

    @classmethod
    async def _play_response_with_tts(
        cls,
        text: str,
        tts_speaker: str | None = None,
        playback_token: int | None = None,
    ):
        """Synthesize text and play it through the speaker."""
        try:
            from core.ref import get_speaker

            resolved_tts_speaker = tts_speaker or cls.get_tts_speaker_for_session_key()
            if resolved_tts_speaker == cls.XIAOAI_TTS_SPEAKER:
                speaker = get_speaker()
                if speaker:
                    await speaker.play(text=text, blocking=True)
                return

            from core.services.tts.doubao import DoubaoTTS

            tts_config = ConfigManager.instance().get_app_config("tts.doubao", {})
            app_id = tts_config.get("app_id")
            access_key = tts_config.get("access_key")
            if not app_id or not access_key:
                logger.warning(
                    "[OpenAI] Doubao TTS credentials not configured, falling back to xiaoai native tts"
                )
                speaker = get_speaker()
                if speaker:
                    await speaker.play(text=text, blocking=True)
                return

            speaker_id = resolved_tts_speaker or tts_config.get(
                "default_speaker", "zh_female_xiaohe_uranus_bigtts"
            )
            tts = DoubaoTTS(
                app_id=app_id,
                access_key=access_key,
                speaker=speaker_id,
            )
            resolved_format = tts.resolve_audio_format(text)
            if tts_config.get("stream", False):
                await open_xiaoai_server.tts_stream_play(
                    text,
                    app_id=app_id,
                    access_key=access_key,
                    resource_id=tts.resource_id,
                    speaker=speaker_id,
                    speed=cls._tts_speed,
                    format=resolved_format,
                    sample_rate=24000,
                    playback_token=playback_token,
                )
            else:
                await open_xiaoai_server.tts_play(
                    text,
                    app_id=app_id,
                    access_key=access_key,
                    resource_id=tts.resource_id,
                    speaker=speaker_id,
                    speed=cls._tts_speed,
                    format=resolved_format,
                    sample_rate=24000,
                    playback_token=playback_token,
                )
        except Exception as exc:
            logger.error(f"[OpenAI] Error playing response with TTS: {exc}")
            try:
                from core.ref import get_speaker

                speaker = get_speaker()
                if speaker:
                    await speaker.play(text=text, blocking=True)
            except Exception as fallback_error:
                logger.error(f"[OpenAI] Fallback TTS also failed: {fallback_error}")
