"""Main application controller.

This module manages the main application flow, coordinating between:
- XiaoAI (Xiaomi speaker service)
- OpenClaw (External integration)
- Home Assistant (Conversation API integration, config-driven — see config.py)
- Audio system (VAD, KWS, Codec)
"""

import asyncio
import os
import threading
import time

from core.xiaoai import XiaoAI
from core.ref import set_app
from core.utils.config import ConfigManager
from core.utils.logger import logger
from core.services.protocols.typing import (
    DeviceState,
    EventType,
)
from core.openclaw import OpenClawManager
from core.services.api_server import APIServer


class MainApp:
    """Main application controller."""

    _instance = None

    @classmethod
    def instance(
        cls,
        enable_openclaw: bool = False,
    ):
        """Get singleton instance.

        Args:
            enable_openclaw: Whether to enable OpenClaw connection (default: False)

        Note:
            Home Assistant is not gated by a flag here — it is enabled purely
            via config.py's ``homeassistant.enabled`` and dispatched through
            WakeupSessionManager (see core/wakeup_session.py).
        """
        if cls._instance is None:
            cls._instance = MainApp(
                enable_openclaw=enable_openclaw,
            )
        return cls._instance

    def __init__(
        self,
        enable_openclaw: bool = False,
    ):
        """Initialize the main application.

        Args:
            enable_openclaw: Whether to enable OpenClaw connection
        """
        if MainApp._instance is not None:
            raise Exception("MainApp is singleton, use instance() to get instance")
        MainApp._instance = self

        # Config
        self.config = ConfigManager.instance()

        # Feature flags
        self._enable_openclaw = enable_openclaw

        # Device state
        self.device_state = DeviceState.IDLE
        self.current_text = ""
        self.current_emotion = "neutral"

        # Event loop and threads
        self.loop = asyncio.new_event_loop()
        self.loop_thread = None
        self.config_watch_thread = None
        self.shutdown_requested = False
        self.running = False

        # Task queue
        self.main_tasks = []
        self.mutex = threading.Lock()

        # Events
        self.events = {
            EventType.SCHEDULE_EVENT: threading.Event(),
        }

        # API Server
        self.api_server = None
        self._enable_api_server = False

        set_app(self)

    def _homeassistant_enabled(self) -> bool:
        """Whether Home Assistant integration is enabled via config.py.

        Home Assistant has no CLI/env feature flag (unlike OpenClaw); it is
        purely config-driven. Any check that needs to know "should the
        wakeup/audio pipeline be running for an external backend" must
        include this, or Home Assistant will silently never receive audio.
        """
        return bool(
            self.config.get_app_config("homeassistant.enabled", False)
        )

    def run(self, enable_api_server: bool = False):
        """Start the main application.

        Args:
            enable_api_server: Whether to start the HTTP API Server
        """
        self._enable_api_server = enable_api_server

        homeassistant_enabled = self._homeassistant_enabled()

        # Check audio input status
        audio_input_enabled = os.environ.get(
            "AUDIO_INPUT_ENABLE", "true"
        ).strip().lower() in ("true", "1", "yes", "on")

        if not audio_input_enabled:
            local_asr_backends = []
            if (
                self._enable_openclaw
                and self.config.get_app_config("openclaw.input_mode", "local_asr")
                == "local_asr"
            ):
                local_asr_backends.append("OpenClaw")
            if (
                homeassistant_enabled
                and self.config.get_app_config("homeassistant.input_mode", "xiaoai_asr")
                == "local_asr"
            ):
                local_asr_backends.append("Home Assistant")
            if local_asr_backends:
                raise RuntimeError(
                    "Audio input is disabled (AUDIO_INPUT_ENABLE=false) but "
                    f"{', '.join(local_asr_backends)} uses 'local_asr' mode. "
                    "Either enable audio input, or set input_mode='xiaoai_asr' in config."
                )

        # Create event loop thread
        self.loop_thread = threading.Thread(target=self._run_event_loop)
        self.loop_thread.daemon = True
        self.loop_thread.start()

        self._start_config_watcher()

        time.sleep(0.1)

        # Initialize XiaoAI service
        asyncio.run_coroutine_threadsafe(XiaoAI.init_xiaoai(), self.loop)

        # Initialize OpenClaw if enabled
        if self._enable_openclaw:
            OpenClawManager.initialize_from_config()
            asyncio.run_coroutine_threadsafe(OpenClawManager.connect(), self.loop)

        # Start API Server if enabled
        if self._enable_api_server:
            host = os.environ.get("API_SERVER_HOST", "127.0.0.1")
            port = int(os.environ.get("API_SERVER_PORT", 9092))
            self.api_server = APIServer(host=host, port=port)
            asyncio.run_coroutine_threadsafe(self.api_server.start(), self.loop)

        # Start main loop thread
        main_loop_thread = threading.Thread(target=self._main_loop)
        main_loop_thread.daemon = True
        main_loop_thread.start()

        # Start audio services
        #
        # IMPORTANT: Home Assistant is config-driven (no enable_* flag), so
        # it must be checked explicitly here — otherwise enabling only
        # Home Assistant (OpenClaw disabled) would never start VAD/KWS and
        # its custom wake words would never fire.
        if self._enable_openclaw or homeassistant_enabled:
            # Check audio input via env var (same as Rust), default True
            # Supports: "true"/"false", "1"/"0", "yes"/"no", "on"/"off"
            audio_input_enabled = os.environ.get(
                "AUDIO_INPUT_ENABLE", "true"
            ).strip().lower() in ("true", "1", "yes", "on")
            if audio_input_enabled:
                from core.services.audio.vad import VAD
                from core.services.audio.kws import KWS
                VAD.start()
                KWS.start()
                logger.info("[MainApp] Audio input enabled (VAD/KWS started)")
            else:
                logger.info("[MainApp] Audio input disabled (VAD/KWS not started)")

            # Pre-warm local ASR only when an enabled backend is configured to use it.
            if audio_input_enabled and (
                (
                    self._enable_openclaw
                    and self.config.get_app_config("openclaw.input_mode", "local_asr")
                    == "local_asr"
                )
                or (
                    homeassistant_enabled
                    and self.config.get_app_config(
                        "homeassistant.input_mode", "xiaoai_asr"
                    )
                    == "local_asr"
                )
            ):
                from core.services.audio.asr import ASRService

                threading.Thread(
                    target=ASRService.ensure_loaded,
                    daemon=True,
                    name="asr-warmup",
                ).start()

    def _run_event_loop(self):
        """Run asyncio event loop in separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def _start_config_watcher(self):
        """Start config file watcher thread."""
        if self.config_watch_thread and self.config_watch_thread.is_alive():
            return

        self.config_watch_thread = threading.Thread(
            target=self._watch_config_file,
            daemon=True,
        )
        self.config_watch_thread.start()

    def _watch_config_file(self):
        """Poll config file for changes and hot-reload."""
        config_path = self.config.get_config_path()
        last_mtime = None

        while True:
            if self.shutdown_requested:
                break

            try:
                current_mtime = os.path.getmtime(config_path)
                if last_mtime is None:
                    last_mtime = current_mtime
                elif current_mtime != last_mtime:
                    last_mtime = current_mtime
                    self.config.reload_app_config()
                    logger.info(f"[Config] Reloaded runtime config from {config_path}")
            except Exception as exc:
                logger.warning(f"[Config] Failed to reload config: {exc}")

            time.sleep(1)

    def _main_loop(self):
        """Main application loop."""
        self.running = True

        while self.running:
            for event_type, event in self.events.items():
                if event.is_set():
                    event.clear()

                    if event_type == EventType.SCHEDULE_EVENT:
                        self._process_scheduled_tasks()

            time.sleep(0.01)

    def _process_scheduled_tasks(self):
        """Process scheduled tasks."""
        with self.mutex:
            tasks = self.main_tasks.copy()
            self.main_tasks.clear()

        for task in tasks:
            try:
                task()
            except Exception as exc:
                logger.error(
                    f"[MainApp] Scheduled task failed: {type(exc).__name__}: {exc}"
                )

    def schedule(self, callback):
        """Schedule task to main loop."""
        with self.mutex:
            if "abort_speaking" in str(callback):
                if any("abort_speaking" in str(task) for task in self.main_tasks):
                    return
            self.main_tasks.append(callback)
        self.events[EventType.SCHEDULE_EVENT].set()

    # State management

    def set_chat_message(self, role, message):
        """Set chat message."""
        self.current_text = message

    def set_emotion(self, emotion):
        """Set emotion."""
        self.current_emotion = emotion

    def alert(self, title, message):
        """Show alert."""
        logger.warning(f"[Alert] {title}: {message}")

    # Shutdown

    def shutdown(self):
        """Shutdown the application."""
        self.shutdown_requested = True
        self.running = False

        if self.api_server:
            asyncio.run_coroutine_threadsafe(
                self.api_server.stop(), self.loop
            )

        # Close OpenClaw connection if connected
        if OpenClawManager.is_connected():
            asyncio.run_coroutine_threadsafe(
                OpenClawManager.close(), self.loop
            )

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=1.0)

        if self.config_watch_thread and self.config_watch_thread.is_alive():
            self.config_watch_thread.join(timeout=1.0)

    # Public API

    async def send_to_openclaw(self, text: str, wait_response: bool = False) -> str | None:
        """Send message to OpenClaw (for skill-based autonomous playback).

        Automatically appends rule_prompt_for_skill if configured.
        Returns run_id or response text on success, None on failure.
        """
        try:
            from core.openclaw import OpenClawManager
            full_text = text
            if OpenClawManager._rule_prompt_for_skill:
                full_text = text + "\n" + OpenClawManager._rule_prompt_for_skill
            return await OpenClawManager.send(full_text, wait_response=wait_response)
        except Exception as e:
            logger.error(f"[MainApp] 发送消息到 OpenClaw 失败: {type(e).__name__}: {e}")
            return None

    async def send_to_openclaw_and_play_reply(self, text: str, wait_response: bool = False) -> str | None:
        """Send message to OpenClaw and play the reply via TTS.

        Automatically appends rule_prompt if configured.
        Returns run_id or response text on success, None on failure.
        """
        try:
            from core.openclaw import OpenClawManager
            full_text = text
            if OpenClawManager._rule_prompt:
                full_text = text + "\n" + OpenClawManager._rule_prompt
            return await OpenClawManager.send_and_play_reply(full_text, wait_response=wait_response)
        except Exception as e:
            logger.error(f"[MainApp] 发送消息到 OpenClaw 失败: {type(e).__name__}: {e}")
            return None

    def set_openclaw_session_key(self, session_key: str):
        """Override the OpenClaw session key at runtime.

        Call this before sending a message or triggering a wakeup to route
        the conversation to a different session.

        Args:
            session_key: New session key (e.g. "agent:user123:my-app").
        """
        OpenClawManager.set_session_key(session_key)
