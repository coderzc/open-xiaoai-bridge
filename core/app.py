"""Main application controller for XiaoZhi Voice Assistant.

This module manages the main application flow, coordinating between:
- XiaoAI (Xiaomi speaker service)
- XiaoZhi (AI conversation service)
- OpenClaw (External integration)
- Audio system (VAD, KWS, Codec)
"""

import asyncio
import json
import threading
import time

from core.xiaozhi import XiaoZhi
from core.xiaoai import XiaoAI
from core.event import EventManager
from core.ref import set_xiaozhi, set_app
from core.services.audio.kws import KWS
from core.services.audio.vad import VAD
from core.utils.config import ConfigManager
from core.utils.logger import logger
from core.services.protocols.typing import (
    AbortReason,
    DeviceState,
    EventType,
)
from config import APP_CONFIG
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

        # Audio
        self.audio_codec = None

        # Event loop and threads
        self.loop = asyncio.new_event_loop()
        self.loop_thread = None
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

        # Create event loop thread
        self.loop_thread = threading.Thread(target=self._run_event_loop)
        self.loop_thread.daemon = True
        self.loop_thread.start()

        time.sleep(0.1)

        # Initialize XiaoAI service
        asyncio.run_coroutine_threadsafe(XiaoAI.init_xiaoai(), self.loop)

        if self._enable_xiaozhi:
            # Create XiaoZhi instance (handles protocol)
            self.xiaozhi = XiaoZhi.instance()
            self.xiaozhi.set_app(self)  # Give xiaozhi reference to app
            set_xiaozhi(self.xiaozhi)   # Register for get_xiaozhi()

            # Initialize XiaoZhi connection
            asyncio.run_coroutine_threadsafe(self._init_xiaozhi(), self.loop)
        else:
            # Without XiaoZhi, still init audio for display/speaker
            self._init_audio()

        # Initialize OpenClaw if enabled
        if self._enable_openclaw:
            # Initialize OpenClaw config from environment variables
            OpenClawManager.initialize_from_config()
            asyncio.run_coroutine_threadsafe(OpenClawManager.connect(), self.loop)

        # Start API Server if enabled
        if self._enable_api_server:
            import os
            api_cfg = APP_CONFIG.get("api_server", {})
            host = os.environ.get("API_SERVER_HOST") or api_cfg.get("host", "127.0.0.1")
            port = int(os.environ.get("API_SERVER_PORT") or api_cfg.get("port", 9092))
            self.api_server = APIServer(host=host, port=port)
            asyncio.run_coroutine_threadsafe(self.api_server.start(), self.loop)

        # Start main loop thread
        main_loop_thread = threading.Thread(target=self._main_loop)
        main_loop_thread.daemon = True
        main_loop_thread.start()

        # Start audio services (only if XiaoZhi enabled, otherwise managed by XiaoAI)
        if self._enable_xiaozhi:
            VAD.start()
            KWS.start()

    def _run_event_loop(self):
        """Run asyncio event loop in separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def _init_xiaozhi(self):
        """Initialize XiaoZhi connection."""
        # Initialize audio codec
        self._init_audio()

        # Set up XiaoZhi callbacks
        self.xiaozhi.on_network_error = self._on_network_error
        self.xiaozhi.on_incoming_audio = self._on_incoming_audio
        self.xiaozhi.on_incoming_json = self._on_incoming_json
        self.xiaozhi.on_audio_channel_opened = self._on_audio_channel_opened
        self.xiaozhi.on_audio_channel_closed = self._on_audio_channel_closed

        # Connect to XiaoZhi server
        self.device_state = DeviceState.CONNECTING
        await self.xiaozhi.connect()

    def _init_audio(self):
        """Initialize audio codec."""
        if not self._enable_xiaozhi:
            return
        try:
            from core.services.audio.codec import AudioCodec

            self.audio_codec = AudioCodec()
            if self.xiaozhi:
                self.xiaozhi.set_audio_codec(self.audio_codec)
        except Exception as e:
            self.alert("Error", f"Failed to initialize audio: {e}")

        # Only start audio trigger if using XiaoZhi mode
        if self._enable_xiaozhi:
            threading.Thread(target=self._audio_input_trigger, daemon=True).start()

    def _main_loop(self):
        """Main application loop."""
        self.running = True

        while self.running:
            for event_type, event in self.events.items():
                if event.is_set():
                    event.clear()

                    if event_type == EventType.AUDIO_INPUT_READY_EVENT:
                        self._handle_input_audio()
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
            except Exception:
                pass

    def schedule(self, callback):
        """Schedule task to main loop."""
        with self.mutex:
            if "abort_speaking" in str(callback):
                if any("abort_speaking" in str(task) for task in self.main_tasks):
                    return
            self.main_tasks.append(callback)
        self.events[EventType.SCHEDULE_EVENT].set()

    def _handle_input_audio(self):
        """Handle audio input."""
        if self.device_state != DeviceState.LISTENING:
            return

        encoded_data = self.audio_codec.read_audio()
        if encoded_data and self.xiaozhi.is_connected():
            asyncio.run_coroutine_threadsafe(
                self.xiaozhi.send_audio(encoded_data), self.loop
            )

    def _audio_input_trigger(self):
        """Trigger audio input event."""
        while self.running:
            if self.audio_codec and self.audio_codec.input_stream.is_active():
                self.events[EventType.AUDIO_INPUT_READY_EVENT].set()
            time.sleep(0.01)

    # Callbacks from XiaoZhi
    def _on_network_error(self, message):
        """Handle network error from XiaoZhi."""
        self.set_device_state(DeviceState.IDLE)
        if self.device_state != DeviceState.CONNECTING:
            self.set_device_state(DeviceState.IDLE)
            if self.xiaozhi:
                asyncio.run_coroutine_threadsafe(
                    self.xiaozhi.disconnect(), self.loop
                )

    def _on_incoming_audio(self, data):
        """Handle incoming audio from XiaoZhi."""
        if self.device_state == DeviceState.SPEAKING:
            self.audio_codec.write_audio(data)

    def _on_incoming_json(self, json_data):
        """Handle incoming JSON from XiaoZhi."""
        try:
            if not json_data:
                return

            data = json.loads(json_data) if isinstance(json_data, str) else json_data
            msg_type = data.get("type", "")

            if msg_type == "tts":
                self._handle_tts_message(data)
            elif msg_type == "stt":
                self._handle_stt_message(data)
            elif msg_type == "llm":
                self._handle_llm_message(data)
        except Exception:
            pass

    def _handle_tts_message(self, data):
        """Handle TTS message."""
        state = data.get("state", "")
        if state == "start":
            EventManager.on_tts_start(data.get("session_id"))
            self.schedule(lambda: self._handle_tts_start())
        elif state == "stop":
            EventManager.on_tts_end(data.get("session_id"))
            self.schedule(lambda: self._handle_tts_stop())
        elif state == "sentence_start":
            text = data.get("text", "")
            if text:
                logger.ai_response(text)
                self.schedule(lambda: self.set_chat_message("assistant", text))

    def _handle_tts_start(self):
        """Handle TTS start."""
        if self.device_state in [DeviceState.IDLE, DeviceState.LISTENING]:
            self.set_device_state(DeviceState.SPEAKING)

    def _handle_tts_stop(self):
        """Handle TTS stop."""
        pass

    def _handle_stt_message(self, data):
        """Handle STT message."""
        text = data.get("text", "")
        if text:
            logger.user_speech(text)
            self.schedule(lambda: self.set_chat_message("user", text))

    def _handle_llm_message(self, data):
        """Handle LLM message."""
        emotion = data.get("emotion", "")
        if emotion:
            self.schedule(lambda: self.set_emotion(emotion))

    def _on_audio_channel_opened(self):
        """Handle audio channel opened."""
        self.set_device_state(DeviceState.IDLE)

    def _on_audio_channel_closed(self):
        """Handle audio channel closed."""
        self.set_device_state(DeviceState.IDLE)
        if self.audio_codec:
            self.audio_codec.stop_streams()

    # Device state management
    def set_device_state(self, state):
        """Set device state."""
        self.device_state = state

        VAD.pause()
        if self.audio_codec:
            self.audio_codec.stop_streams()

        if state == DeviceState.IDLE:
            pass
        elif state == DeviceState.CONNECTING:
            pass
        elif state == DeviceState.LISTENING:
            if self.audio_codec:
                if self.audio_codec.output_stream.is_active():
                    self.audio_codec.output_stream.stop_stream()
                if not self.audio_codec.input_stream.is_active():
                    self.audio_codec.input_stream.start_stream()
        elif state == DeviceState.SPEAKING:
            if self.audio_codec:
                if self.audio_codec.input_stream.is_active():
                    self.audio_codec.input_stream.stop_stream()
                if not self.audio_codec.output_stream.is_active():
                    self.audio_codec.output_stream.start_stream()

    def set_chat_message(self, role, message):
        """Set chat message."""
        self.current_text = message

    def set_emotion(self, emotion):
        """Set emotion."""
        self.current_emotion = emotion

    # User actions
    def start_listening(self):
        """Start listening."""
        self.schedule(self._start_listening_impl)

    def _start_listening_impl(self):
        """Start listening implementation."""
        if not self.xiaozhi:
            return

        self.set_device_state(DeviceState.IDLE)
        asyncio.run_coroutine_threadsafe(
            self.xiaozhi.send_abort_speaking(AbortReason.ABORT),
            self.loop,
        )
        from core.services.protocols.typing import ListeningMode
        asyncio.run_coroutine_threadsafe(
            self.xiaozhi.send_start_listening(ListeningMode.MANUAL), self.loop
        )
        self.set_device_state(DeviceState.LISTENING)

    def stop_listening(self):
        """Stop listening."""
        self.schedule(self._stop_listening_impl)

    def _stop_listening_impl(self):
        """Stop listening implementation."""
        if not self.xiaozhi:
            return
        asyncio.run_coroutine_threadsafe(
            self.xiaozhi.send_stop_listening(), self.loop
        )
        self.set_device_state(DeviceState.IDLE)

    def abort_speaking(self, reason):
        """Abort speaking."""
        if not self.xiaozhi:
            return
        self.set_device_state(DeviceState.IDLE)
        asyncio.run_coroutine_threadsafe(
            self.xiaozhi.send_abort_speaking(AbortReason.ABORT),
            self.loop,
        )

    def alert(self, title, message):
        """Show alert."""
        logger.warning(f"[Alert] {title}: {message}")

    # Shutdown
    def shutdown(self):
        """Shutdown the application."""
        self.running = False

        if self.audio_codec:
            self.audio_codec.close()

        if self.xiaozhi:
            asyncio.run_coroutine_threadsafe(
                self.xiaozhi.disconnect(), self.loop
            )

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

    # Public API
    async def send_text(self, text):
        """Send text to XiaoZhi."""
        if self.xiaozhi and self.xiaozhi.is_connected():
            await self.xiaozhi.send_text(text)

    async def send_to_openclaw(self, text: str, wait_response: bool = False) -> bool | str | None:
        """Send message to OpenClaw."""
        try:
            return await OpenClawManager.send_message(text, wait_response=wait_response)
        except Exception as e:
            logger.error(f"[MainApp] 发送消息到 OpenClaw 失败: {type(e).__name__}: {e}")
            return False
