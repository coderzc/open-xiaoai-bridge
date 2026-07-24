"""QwenPaw service manager.

This module talks to a running QwenPaw application through its HTTP
console task API and exposes the same minimal interface used by the
external conversation controller.
"""

import asyncio
import uuid
from typing import Any

import aiohttp

import open_xiaoai_server

from core.utils.base import get_env
from core.utils.config import ConfigManager
from core.utils.logger import logger


class QwenPawManager:
    """Manager for QwenPaw HTTP chat backend."""

    XIAOAI_TTS_SPEAKER = "xiaoai"

    _initialized = False
    _reload_listener_registered = False
    _enabled = False
    _base_url = "http://127.0.0.1:8088"
    _agent_id = "default"
    _user_id = "open-xiaoai-bridge"
    _session_key = "open-xiaoai-bridge"
    _send_path = "/api/console/chat/task"
    _task_status_path = "/api/console/chat/task/{task_id}"
    _response_timeout = 120
    _poll_interval = 0.5
    _tts_speaker = None
    _session_tts_speakers: dict[str, str] = {}
    _tts_speed = 1.0
    _rule_prompt = ""
    _rule_prompt_for_skill = ""
    _auth_token = ""
    _auth_header = "Authorization"
    _auth_scheme = "Bearer"
    _extra_headers: dict[str, str] = {}
    _response_events: dict[str, asyncio.Future] = {}
    _response_texts: dict[str, str] = {}
    _response_tts_speakers: dict[str, str | None] = {}
    last_error: str | None = None

    @classmethod
    def initialize_from_config(cls, enabled: bool | None = None):
        logger.info("[QwenPaw] Initializing from config...")
        cls.reload_from_config(enabled=enabled)
        cls._initialized = True

    @classmethod
    def reload_from_config(cls, enabled: bool | None = None):
        """Refresh QwenPaw settings from config.py."""
        config_manager = ConfigManager.instance()
        if not cls._reload_listener_registered:
            config_manager.add_reload_listener(
                lambda _old, _new: cls.reload_from_config()
            )
            cls._reload_listener_registered = True

        config = config_manager.get_app_config("qwenpaw", {})

        if enabled is not None:
            cls._enabled = enabled
        else:
            env_enabled = get_env("QWENPAW_ENABLE")
            cls._enabled = (
                env_enabled.lower() in ("1", "true", "yes")
                if env_enabled is not None
                else False
            )

        cls._base_url = str(config.get("base_url", "http://127.0.0.1:8088")).rstrip("/")
        cls._agent_id = str(config.get("agent_id", "default") or "default")
        cls._user_id = str(config.get("user_id", "open-xiaoai-bridge") or "open-xiaoai-bridge")
        cls._session_key = str(config.get("session_key", "open-xiaoai-bridge") or "open-xiaoai-bridge")
        cls._send_path = str(config.get("send_path", "/api/console/chat/task") or "/api/console/chat/task")
        cls._task_status_path = str(
            config.get("task_status_path", "/api/console/chat/task/{task_id}")
            or "/api/console/chat/task/{task_id}"
        )
        cls._response_timeout = int(config.get("response_timeout", 120))
        cls._poll_interval = max(0.1, float(config.get("poll_interval", 0.5)))
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
        cls._auth_token = str(config.get("auth_token", "") or "")
        cls._auth_header = str(config.get("auth_header", "Authorization") or "Authorization")
        cls._auth_scheme = str(config.get("auth_scheme", "Bearer") or "").strip()
        cls._extra_headers = (
            {
                str(key): str(value)
                for key, value in config.get("extra_headers", {}).items()
                if key and value is not None
            }
            if isinstance(config.get("extra_headers", {}), dict)
            else {}
        )

        if cls._enabled:
            logger.info(
                f"[QwenPaw] Enabled, base_url={cls._base_url}, agent_id={cls._agent_id}"
            )

    @classmethod
    async def connect(cls) -> bool:
        """No-op connect hook for parity with other backends."""
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
            f"[QwenPaw] Session key updated: {cls._session_key!r} -> {session_key!r}"
        )
        cls._session_key = session_key

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
            logger.warning("[QwenPaw] send called but backend is disabled")
            return None

        run_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        cls._response_events[run_id] = loop.create_future()
        cls._response_texts[run_id] = ""
        cls._response_tts_speakers[run_id] = cls.get_tts_speaker_for_session_key()
        logger.user_speech(text, module=f"QwenPaw({cls._session_key})")
        asyncio.create_task(cls._run_qwenpaw_task(run_id, text))
        return run_id

    @classmethod
    async def _wait_response(cls, run_id: str) -> str | None:
        event = cls._response_events.get(run_id)
        if not event:
            logger.warning(f"[QwenPaw] No event found for run {run_id}")
            return None
        try:
            await asyncio.wait_for(event, timeout=cls._response_timeout)
            return cls._response_texts.pop(run_id, "") or None
        except asyncio.TimeoutError:
            logger.warning(f"[QwenPaw] Timeout waiting for response (runId: {run_id})")
            return None
        finally:
            cls._response_events.pop(run_id, None)
            cls._response_texts.pop(run_id, None)

    @classmethod
    async def _run_qwenpaw_task(cls, run_id: str, text: str):
        try:
            response_text = await cls._request_qwenpaw(text)
            if response_text:
                cls._response_texts[run_id] = response_text
                logger.ai_response(response_text, module=f"QwenPaw({cls._session_key})")
        except Exception as exc:
            cls.last_error = f"{type(exc).__name__}: {exc}"
            logger.error(f"[QwenPaw] Request failed: {cls.last_error}")
        finally:
            waiter = cls._response_events.get(run_id)
            if waiter and not waiter.done():
                waiter.get_loop().call_soon_threadsafe(waiter.set_result, None)

    @classmethod
    async def _request_qwenpaw(cls, text: str) -> str | None:
        timeout = aiohttp.ClientTimeout(total=cls._response_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                cls._url_for_path(cls._send_path),
                json=cls._build_payload(text),
                headers=cls._headers(),
            ) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {body}")

            task_id = body.get("task_id") if isinstance(body, dict) else None
            if not task_id:
                return cls._extract_response_text(body)

            return await cls._poll_task_result(session, str(task_id))

    @classmethod
    def _build_payload(cls, text: str) -> dict[str, Any]:
        return {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }
            ],
            "session_id": cls._session_key,
            "user_id": cls._user_id,
            "stream": True,
            "timeout": cls._response_timeout,
        }

    @classmethod
    def _headers(cls) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if cls._agent_id:
            headers["X-Agent-Id"] = cls._agent_id
        if cls._auth_token:
            if cls._auth_scheme:
                headers[cls._auth_header] = f"{cls._auth_scheme} {cls._auth_token}"
            else:
                headers[cls._auth_header] = cls._auth_token
        headers.update(cls._extra_headers)
        return headers

    @classmethod
    async def _poll_task_result(
        cls,
        session: aiohttp.ClientSession,
        task_id: str,
    ) -> str | None:
        deadline = asyncio.get_running_loop().time() + cls._response_timeout
        while True:
            if asyncio.get_running_loop().time() >= deadline:
                raise asyncio.TimeoutError(f"QwenPaw task timeout: {task_id}")

            async with session.get(
                cls._url_for_path(cls._task_status_path.format(task_id=task_id)),
                headers=cls._headers(),
            ) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {body}")

            status = body.get("status") if isinstance(body, dict) else None
            if status == "finished":
                result = body.get("result") if isinstance(body, dict) else body
                return cls._extract_response_text(result)
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"QwenPaw task {status}: {body}")
            await asyncio.sleep(cls._poll_interval)

    @classmethod
    def _url_for_path(cls, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return cls._base_url + (path if path.startswith("/") else f"/{path}")

    @classmethod
    def _extract_response_text(cls, body: Any) -> str | None:
        texts: list[str] = []
        cls._collect_response_texts(body, texts)
        return "".join(texts).strip() or None

    @classmethod
    def _collect_response_texts(cls, value: Any, texts: list[str]):
        if isinstance(value, str):
            return
        if isinstance(value, list):
            for item in value:
                cls._collect_response_texts(item, texts)
            return
        if not isinstance(value, dict):
            return

        value_type = value.get("type")
        if value_type in {"turn_usage", "reasoning", "progress"}:
            return

        role = value.get("role")
        if role and role not in {"assistant", "system"}:
            return

        content = value.get("content")
        if content is not None:
            cls._collect_response_texts(content, texts)

        text = value.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)

        output = value.get("output")
        if output is not None:
            cls._collect_response_texts(output, texts)

        result = value.get("result")
        if result is not None:
            cls._collect_response_texts(result, texts)

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
                logger.warning(f"[QwenPaw] No response text received for run {run_id}")
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
                    "[QwenPaw] Doubao TTS credentials not configured, falling back to xiaoai native tts"
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
            logger.error(f"[QwenPaw] Error playing response with TTS: {exc}")
            try:
                from core.ref import get_speaker

                speaker = get_speaker()
                if speaker:
                    await speaker.play(text=text, blocking=True)
            except Exception as fallback_error:
                logger.error(f"[QwenPaw] Fallback TTS also failed: {fallback_error}")
