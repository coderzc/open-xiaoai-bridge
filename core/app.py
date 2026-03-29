"""Main application controller.

This module manages the main application flow, coordinating between:
- XiaoAI (Xiaomi speaker service)
- XiaoZhi (AI conversation service)
- OpenClaw (External integration)
- Audio system (VAD, KWS, Codec)
"""

import asyncio
import os
import threading
import time

from core.xiaozhi import XiaoZhi
from core.xiaoai import XiaoAI
from core.ref import set_xiaozhi, set_app
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
    def instance(cls, enable_xiaozhi: bool = True, enable_openclaw: bool = False):
        """Get singleton instance.

        Args:
            enable_xiaozhi: Whether to enable XiaoZhi AI connection (default: True)
            enable_openclaw: Whether to enable OpenClaw connection (default: False)
        """
        if cls._instance is None:
            cls._instance = MainApp(enable_xiaozhi=enable_xiaozhi, enable_openclaw=enable_openclaw)
        return cls._instance

    def __init__(self, enable_xiaozhi: bool = True, enable_openclaw: bool = False):
        """Initialize the main application.

        Args:
            enable_xiaozhi: Whether to enable XiaoZhi AI connection
            enable_openclaw: Whether to enable OpenClaw connection
        """
        if MainApp._instance is not None:
            raise Exception("MainApp is singleton, use instance() to get instance")
        MainApp._instance = self

        # Config
        self.config = ConfigManager.instance()

        # Feature flags
        self._enable_xiaozhi = enable_xiaozhi
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
            EventType.AUDIO_INPUT_READY_EVENT: threading.Event(),
        }

        # XiaoZhi instance (protocol layer)
        self.xiaozhi = None

        # API Server
        self.api_server = None
        self._enable_api_server = False

        set_app(self)

    @property
    def protocol(self):
        """Access XiaoZhi protocol for backward compatibility."""
        if self.xiaozhi:
            return self.xiaozhi.protocol
        return None

    def run(self, enable_api_server: bool = False):
        """Start the main application.

        Args:
            enable_api_server: Whether to start the HTTP API Server
        """
        self._enable_api_server = enable_api_server

        # Check audio input status
        audio_input_enabled = os.environ.get(
            "AUDIO_INPUT_ENABLE", "true"
        ).strip().lower() in ("true", "1", "yes", "on")
        
        if not audio_input_enabled and self._enable_xiaozhi:
            raise RuntimeError(
                "Audio input is disabled (AUDIO_INPUT_ENABLE=false) but XiaoZhi is enabled. "
                "Either enable audio input or disable XiaoZhi."
            )
        
        if not audio_input_enabled and self._enable_openclaw:
            openclaw_input_mode = self.config.get_app_config("openclaw.input_mode", "local_asr")
            if openclaw_input_mode == "local_asr":
                raise RuntimeError(
                    "Audio input is disabled (AUDIO_INPUT_ENABLE=false) but OpenClaw uses 'local_asr' mode. "
                    "Either enable audio input, or set openclaw.input_mode='xiaoai_asr' in config."
                )

        # Create event loop thread
        self.loop_thread = threading.Thread(target=self._run_event_loop)
        self.loop_thread.daemon = True
        self.loop_thread.start()

        self._start_config_watcher()

        time.sleep(0.1)

        # Initialize XiaoAI service
        asyncio.run_coroutine_threadsafe(XiaoAI.init_xiaoai(), self.loop)

        if self._enable_xiaozhi:
            # Create XiaoZhi instance
            self.xiaozhi = XiaoZhi.instance()
            self.xiaozhi.set_app(self)
            set_xiaozhi(self.xiaozhi)

            # Initialize XiaoZhi connection
            asyncio.run_coroutine_threadsafe(self._init_xiaozhi(), self.loop)

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
        if self._enable_xiaozhi or self._enable_openclaw:
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

            # Pre-warm local ASR only when OpenClaw is configured to use it
            # and audio input is enabled.
            if audio_input_enabled and self._enable_openclaw:
                input_mode = self.config.get_app_config(
                    "openclaw.input_mode",
                    "local_asr",
                )
                if input_mode == "local_asr":
                    from core.services.audio.asr import ASRManager

                    threading.Thread(
                        target=ASRManager.warmup,
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

    async def _init_xiaozhi(self):
        """Initialize XiaoZhi connection and audio."""
        self.device_state = DeviceState.CONNECTING
        await self.xiaozhi.connect()
        self.xiaozhi.init_audio()

    def _main_loop(self):
        """Main application loop."""
        self.running = True

        while self.running:
            for event_type, event in self.events.items():
                if event.is_set():
                    event.clear()

                    if event_type == EventType.AUDIO_INPUT_READY_EVENT:
                        if self.xiaozhi:
                            self.xiaozhi.handle_input_audio()
                    elif event_type == EventType.SCHEDULE_EVENT:
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

        if self.xiaozhi:
            self.xiaozhi.shutdown()

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

    async def send_text(self, text):
        """Send text to XiaoZhi."""
        if self.xiaozhi and self.xiaozhi.is_connected():
            await self.xiaozhi.send_text(text)

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
