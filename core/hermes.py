"""基于 OpenAI-compatible 传输的 Hermes Agent 专用后端。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import inspect
import re
from typing import Any

import aiohttp

from core.hermes_progress import HermesToolProgress
from core.openai import OpenAIManager
from core.openai_stream import stream_openai_chat_completion
from core.utils.base import get_env
from core.utils.config import ConfigManager
from core.utils.logger import logger


class HermesManager(OpenAIManager):
    """使用独立配置与会话状态的 Hermes 专用后端。"""

    CONFIG_PREFIX = "hermes"
    ENV_ENABLE = "HERMES_ENABLE"
    LOG_NAME = "Hermes"
    DEFAULT_BASE_URL = "http://127.0.0.1:8642/v1"
    DEFAULT_MODEL = "hermes-agent"
    DEFAULT_SESSION_KEY = "agent:default:open-xiaoai-bridge"
    DEFAULT_SESSION_HEADER = "X-Hermes-Session-Key"
    DEFAULT_RESPONSE_MODE = "streaming"
    RESPONSE_MODES = frozenset({"streaming", "complete"})
    CAPABILITIES_TIMEOUT = 2.0
    _PROFILE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
    SESSION_ID_HEADER = "X-Hermes-Session-Id"

    # Hermes 和普通 OpenAI 可同时启用，因此可变运行状态必须由子类独立持有。
    _initialized = False
    _reload_listener_registered = False
    _enabled = False
    _base_url = DEFAULT_BASE_URL
    _api_key = ""
    _model = DEFAULT_MODEL
    _session_key = DEFAULT_SESSION_KEY
    _session_header = DEFAULT_SESSION_HEADER
    _system_prompt = ""
    _temperature = None
    _max_tokens = None
    _timeout = 120
    _history_max_messages = 20
    _extra_body: dict[str, Any] = {}
    _tts_provider: str | None = None
    _tts_speaker = None
    _session_tts_speakers: dict[str, str] = {}
    _tts_speed = 1.0
    _rule_prompt = ""
    _rule_prompt_for_skill = ""
    _response_mode = DEFAULT_RESPONSE_MODE
    _profile = ""
    _profile_api_keys: dict[str, str] = {}
    _sessions: dict[str, list[dict[str, str]]] = {}
    _hermes_session_ids: dict[str, str] = {}
    _capabilities_checked = False
    _capabilities: dict[str, Any] | None = None
    _capabilities_diagnostic: str | None = None
    _response_events: dict[str, asyncio.Future] = {}
    _response_texts: dict[str, str] = {}
    _response_tts_speakers: dict[str, str | None] = {}
    last_error: str | None = None

    @classmethod
    def initialize_from_config(cls, enabled: bool | None = None):
        logger.info("[Hermes] Initializing from config...")
        cls.reload_from_config(enabled=enabled)
        cls._initialized = True

    @classmethod
    def reload_from_config(cls, enabled: bool | None = None):
        """刷新 Hermes 配置，不触碰普通 OpenAI 的状态。"""

        config_manager = ConfigManager.instance()
        if not cls._reload_listener_registered:
            config_manager.add_reload_listener(
                lambda _old, _new: cls.reload_from_config()
            )
            cls._reload_listener_registered = True

        config = config_manager.get_app_config(cls.CONFIG_PREFIX, {})
        if enabled is not None:
            cls._enabled = enabled
        else:
            env_enabled = get_env(cls.ENV_ENABLE)
            cls._enabled = (
                env_enabled.lower() in ("1", "true", "yes")
                if env_enabled is not None
                else False
            )

        cls._base_url = str(
            config.get("base_url", cls.DEFAULT_BASE_URL)
        ).rstrip("/")
        cls._api_key = str(config.get("api_key", "") or "")
        cls._model = str(config.get("model", cls.DEFAULT_MODEL))
        cls._session_key = str(
            config.get("session_key", cls.DEFAULT_SESSION_KEY)
        )
        cls._session_header = str(
            config.get("session_header", cls.DEFAULT_SESSION_HEADER) or ""
        ).strip()
        cls._system_prompt = str(config.get("system_prompt", "") or "")
        cls._timeout = int(config.get("response_timeout", 120))
        cls._history_max_messages = max(
            0,
            int(config.get("history_max_messages", 20)),
        )
        cls._temperature = cls._optional_float(config.get("temperature"))
        cls._max_tokens = cls._optional_int(config.get("max_tokens"))
        cls._extra_body = config.get("extra_body", {})
        if not isinstance(cls._extra_body, dict):
            cls._extra_body = {}
        configured_provider = config.get("tts_provider")
        cls._tts_provider = (
            str(configured_provider).strip().lower()
            if configured_provider
            else None
        )
        cls._tts_speaker = config.get("tts_speaker", None)
        cls._session_tts_speakers = (
            {
                str(key): str(value)
                for key, value in config.get(
                    "session_tts_speakers",
                    {},
                ).items()
                if key and value
            }
            if isinstance(config.get("session_tts_speakers", {}), dict)
            else {}
        )
        cls._tts_speed = float(config.get("tts_speed", 1.0))
        cls._rule_prompt = str(config.get("rule_prompt", "") or "")
        cls._rule_prompt_for_skill = str(
            config.get("rule_prompt_for_skill", "") or ""
        )
        response_mode = str(
            config.get("response_mode", cls.DEFAULT_RESPONSE_MODE) or ""
        ).strip().lower()
        if response_mode not in cls.RESPONSE_MODES:
            logger.warning(
                f"[Hermes] Unknown response_mode={response_mode!r}; "
                f"using {cls.DEFAULT_RESPONSE_MODE!r}"
            )
            response_mode = cls.DEFAULT_RESPONSE_MODE
        cls._response_mode = response_mode
        cls._profile_api_keys = (
            {
                str(key): str(value)
                for key, value in config.get("profile_api_keys", {}).items()
                if key and value
            }
            if isinstance(config.get("profile_api_keys", {}), dict)
            else {}
        )
        configured_profile = str(config.get("profile", "") or "").strip()
        cls._profile = cls._validate_profile(configured_profile)

        if cls._enabled:
            logger.info(
                f"[Hermes] Enabled, base_url={cls._base_url}, "
                f"model={cls._model}, response_mode={cls._response_mode}"
            )

    @classmethod
    def get_response_mode(cls) -> str:
        """Return the explicit final-response delivery mode."""

        return cls._response_mode

    @classmethod
    def set_session_key(cls, session_key: str):
        logger.info(
            f"[Hermes] Session key updated: "
            f"{cls._session_key!r} -> {session_key!r}"
        )
        cls._session_key = session_key

    @classmethod
    def get_session_key(cls) -> str:
        """Return the active Hermes long-term-memory scope key."""

        return cls._session_key

    @classmethod
    def set_profile(cls, profile: str | None):
        """Route subsequent requests through Hermes' native profile prefix."""

        normalized = cls._validate_profile(profile)
        if normalized and normalized != "default":
            if normalized not in cls._profile_api_keys:
                raise ValueError(
                    f"No API key configured for Hermes profile {normalized!r}"
                )
        logger.info(
            f"Hermes profile updated: {cls.get_profile()!r} -> "
            f"{normalized or 'default'!r}",
            module="Hermes",
        )
        cls._profile = normalized

    @classmethod
    def get_profile(cls) -> str:
        return cls._profile or "default"

    @classmethod
    def get_session_state(cls) -> dict[str, Any]:
        """Return lightweight, non-secret state for the active profile/session."""

        scope_key = cls._conversation_scope_key()
        return {
            "profile": cls.get_profile(),
            "session_key": cls._session_key,
            "session_id": cls._hermes_session_ids.get(scope_key),
            "history_messages": len(cls._sessions.get(scope_key, [])),
        }

    @classmethod
    def reset_session(cls, session_key: str | None = None):
        """Discard both Bridge text history and Hermes' native transcript."""

        target_session_key = session_key or cls._session_key
        scope_key = cls._conversation_scope_key(target_session_key)
        cls._sessions.pop(scope_key, None)
        cls._hermes_session_ids.pop(scope_key, None)

    @classmethod
    def _validate_profile(cls, profile: str | None) -> str:
        normalized = str(profile or "").strip()
        if not normalized or normalized == "default":
            return ""
        if not cls._PROFILE_RE.fullmatch(normalized):
            raise ValueError(
                "Hermes profile must match "
                "^[a-z0-9][a-z0-9_-]{0,63}$"
            )
        return normalized

    @classmethod
    def _conversation_scope_key(
        cls,
        session_key: str | None = None,
    ) -> str:
        return f"{cls.get_profile()}\x1f{session_key or cls._session_key}"

    @classmethod
    def _capture_response_headers(
        cls,
        headers,
        *,
        session_key: str,
    ):
        session_id = str(
            headers.get(cls.SESSION_ID_HEADER, "") or ""
        ).strip()
        if not session_id:
            return
        cls._hermes_session_ids[session_key] = session_id
        logger.debug(
            f"Captured native session id for {session_key!r}",
            module="Hermes",
        )

    @classmethod
    def _headers(cls) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = (
            cls._profile_api_keys.get(cls._profile, "")
            if cls._profile
            else cls._api_key
        )
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if cls._session_header and cls._session_key:
            headers[cls._session_header] = cls._session_key
        session_id = cls._hermes_session_ids.get(
            cls._conversation_scope_key()
        )
        if session_id:
            headers[cls.SESSION_ID_HEADER] = session_id
        return headers

    @classmethod
    def _profiled_base_url(cls) -> str:
        if not cls._profile:
            return cls._base_url

        marker = "/v1"
        marker_index = cls._base_url.find(marker)
        if marker_index < 0:
            return (
                cls._base_url.rstrip("/")
                + f"/p/{cls._profile}/v1"
            )

        prefix = cls._base_url[:marker_index].rstrip("/")
        suffix = cls._base_url[marker_index:]
        profile_marker = re.search(r"/p/[a-z0-9][a-z0-9_-]{0,63}$", prefix)
        if profile_marker:
            prefix = prefix[:profile_marker.start()]
        return f"{prefix}/p/{cls._profile}{suffix}"

    @classmethod
    def _chat_completions_url(cls) -> str:
        base_url = cls._profiled_base_url()
        if base_url.endswith("/chat/completions"):
            return base_url
        return base_url.rstrip("/") + "/chat/completions"

    @classmethod
    def _capabilities_url(cls) -> str:
        base_url = cls._profiled_base_url().rstrip("/")
        if base_url.endswith("/chat/completions"):
            base_url = base_url.removesuffix("/chat/completions")
        if base_url.endswith("/v1"):
            return base_url + "/capabilities"
        return base_url + "/v1/capabilities"

    @classmethod
    async def connect(cls) -> bool:
        """Probe Hermes metadata once without making startup depend on it."""

        if not cls._initialized:
            cls.initialize_from_config()
        if not cls._enabled:
            return False
        await cls._probe_capabilities_once()
        return True

    @classmethod
    async def _probe_capabilities_once(cls) -> dict[str, Any] | None:
        if cls._capabilities_checked:
            return cls._capabilities
        cls._capabilities_checked = True

        headers = {
            key: value
            for key, value in cls._headers().items()
            if key == "Authorization"
        }
        try:
            timeout = aiohttp.ClientTimeout(total=cls.CAPABILITIES_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    cls._capabilities_url(),
                    headers=headers,
                ) as response:
                    if response.status == 404:
                        cls._capabilities_diagnostic = "unsupported (HTTP 404)"
                        logger.info(
                            "Capabilities endpoint is unavailable; continuing "
                            "with Chat Completions",
                            module="Hermes",
                        )
                        return None
                    if response.status >= 400:
                        cls._capabilities_diagnostic = (
                            f"HTTP {response.status}"
                        )
                        logger.warning(
                            "Capabilities probe failed with "
                            f"HTTP {response.status}; continuing with Chat Completions",
                            module="Hermes",
                        )
                        return None
                    body = await response.json(content_type=None)
        except asyncio.TimeoutError:
            cls._capabilities_diagnostic = "timeout"
            logger.warning(
                "Capabilities probe timed out; continuing with Chat Completions",
                module="Hermes",
            )
            return None
        except Exception as exc:
            cls._capabilities_diagnostic = type(exc).__name__
            logger.warning(
                "Capabilities probe failed; continuing with Chat Completions: "
                f"{type(exc).__name__}: {exc}",
                module="Hermes",
            )
            return None

        if not isinstance(body, dict) or (
            body.get("object") != "hermes.api_server.capabilities"
            or body.get("platform") != "hermes-agent"
        ):
            cls._capabilities_diagnostic = "malformed response"
            logger.warning(
                "Capabilities response is not a Hermes API capability object; "
                "continuing with Chat Completions",
                module="Hermes",
            )
            return None

        cls._capabilities = body
        cls._capabilities_diagnostic = "available"
        endpoints = body.get("endpoints", {})
        surfaces = sorted(endpoints) if isinstance(endpoints, dict) else []
        logger.info(
            "Hermes-aware endpoint detected; available surfaces: "
            + (", ".join(surfaces) if surfaces else "not advertised"),
            module="Hermes",
        )
        return body

    @classmethod
    def get_capabilities(cls) -> dict[str, Any] | None:
        """Return discovered capabilities without triggering another request."""

        return dict(cls._capabilities) if cls._capabilities else None

    @classmethod
    async def _request_chat_completion(cls, text: str) -> str | None:
        """Send a non-streaming turn while retaining Hermes session state."""

        session_key = cls._session_key
        scope_key = cls._conversation_scope_key(session_key)
        history = cls._sessions.setdefault(scope_key, [])
        payload: dict[str, Any] = {
            "model": cls._model,
            "messages": cls._build_messages(history, text),
            "stream": False,
            **cls._extra_body,
        }
        if cls._temperature is not None:
            payload["temperature"] = cls._temperature
        if cls._max_tokens is not None:
            payload["max_tokens"] = cls._max_tokens

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=cls._timeout)
        ) as session:
            async with session.post(
                cls._chat_completions_url(),
                json=payload,
                headers=cls._headers(),
            ) as response:
                body = await response.json(content_type=None)
                if response.status >= 400:
                    message = (
                        body.get("error", body)
                        if isinstance(body, dict)
                        else body
                    )
                    raise RuntimeError(
                        f"HTTP {response.status}: {message}"
                    )
                cls._capture_response_headers(
                    response.headers,
                    session_key=scope_key,
                )

        response_text = cls._extract_response_text(body)
        if response_text:
            cls._append_history(history, text, response_text)
        return response_text

    @classmethod
    async def request_streaming_chat_completion(
        cls,
        text: str,
        *,
        on_delta: Callable[[str], Awaitable[None] | None],
        on_tool_progress: (
            Callable[[HermesToolProgress], Awaitable[None] | None]
            | None
        ) = None,
    ) -> str:
        """流式接收标准文本和 Hermes 工具生命周期事件。"""

        async def on_event(event):
            if event.event != "hermes.tool.progress":
                logger.debug(
                    f"Ignoring unsupported SSE event: {event.event}",
                    module="Hermes",
                )
                return
            try:
                progress = HermesToolProgress.from_json(event.data)
            except Exception as exc:
                logger.debug(
                    f"Ignoring malformed tool progress event: {exc}",
                    module="Hermes",
                )
                return
            if not progress:
                return
            logger.debug(
                f"Tool progress event: {progress.technical_log()}",
                module="Hermes",
            )
            if on_tool_progress:
                result = on_tool_progress(progress)
                if inspect.isawaitable(result):
                    await result

        return await stream_openai_chat_completion(
            cls,
            text,
            on_delta=on_delta,
            on_event=on_event,
            log_name="Hermes",
        )

    @classmethod
    async def _play_response_with_tts(
        cls,
        text: str,
        tts_speaker: str | None = None,
        playback_token: int | None = None,
    ) -> None:
        """等待共享 Router 处理 TTS；沿用上游不返回播放结果的契约。"""
        from core.services.tts.router import TTSRouter

        await TTSRouter.play(
            text,
            configured_provider=cls._tts_provider,
            tts_speaker=tts_speaker or cls.get_tts_speaker_for_session_key(),
            tts_speed=cls._tts_speed,
            playback_token=playback_token,
            log_prefix="Hermes",
        )
