"""XiaoZhi protocol client - handles WebSocket connection to XiaoZhi server.

This module is responsible for:
- Connecting to XiaoZhi WebSocket server
- Sending/receiving audio and text messages
- Protocol-level message handling
- Audio codec management and stream control
- JSON message parsing (TTS/STT/LLM)
- Wakeup session: VAD-driven listen/stop cycle

The main application flow is managed by app.py.
"""

import asyncio
import json
import os
import threading

import open_xiaoai_server

from core.ref import get_audio_codec, get_speaker, get_vad, set_speech_frames
from core.services.protocols.protocol import Protocol
from core.services.protocols.typing import (
    AbortReason,
    DeviceState,
    ListeningMode,
)
from core.utils.config import ConfigManager
from core.utils.logger import logger

_NOTIFY_SOUND_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "assets", "sounds", "tts_notify.mp3",
)


def _load_notify_sound() -> bytes | None:
    """Decode tts_notify.mp3 to PCM at startup."""
    if not os.path.isfile(_NOTIFY_SOUND_PATH):
        return None
    try:
        with open(_NOTIFY_SOUND_PATH, "rb") as f:
            mp3_data = f.read()
        return open_xiaoai_server.decode_audio(mp3_data, format="mp3", sample_rate=24000)
    except Exception:
        return None


_NOTIFY_PCM = _load_notify_sound()


class XiaoZhi:
    """XiaoZhi protocol client."""

    _instance = None

    @classmethod
    def instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = XiaoZhi()
        return cls._instance

    def __init__(self):
        """Initialize XiaoZhi client."""
        if XiaoZhi._instance is not None:
            raise Exception("XiaoZhi is singleton, use instance() to get instance")
        XiaoZhi._instance = self

        self.config = ConfigManager.instance()

        # Protocol
        self.protocol = None

        # References (set by MainApp)
        self._app = None
        self._audio_codec = None

        # Wakeup session state
        self._session_loop: asyncio.AbstractEventLoop | None = None
        self._vad_future: asyncio.Future | None = None
        self._tts_stop_future: asyncio.Future | None = None
        self._is_first_round = False

    def set_app(self, app):
        """Set reference to main app."""
        self._app = app

    def set_audio_codec(self, codec):
        """Set audio codec reference."""
        self._audio_codec = codec

    @property
    def device_state(self):
        """Proxy device state from app."""
        if self._app:
            return self._app.device_state
        return None

    def set_device_state(self, state):
        """Set device state and manage audio streams accordingly."""
        if not self._app:
            return
        self._app.device_state = state

        from core.services.audio.vad import VAD
        VAD.pause()

        if self._audio_codec:
            self._audio_codec.stop_streams()

        if state == DeviceState.LISTENING:
            if self._audio_codec:
                if self._audio_codec.output_stream.is_active():
                    self._audio_codec.output_stream.stop_stream()
                if not self._audio_codec.input_stream.is_active():
                    self._audio_codec.input_stream.start_stream()
        elif state == DeviceState.SPEAKING:
            if self._audio_codec:
                if self._audio_codec.input_stream.is_active():
                    self._audio_codec.input_stream.stop_stream()
                if not self._audio_codec.output_stream.is_active():
                    self._audio_codec.output_stream.start_stream()

    # Connection management

    async def connect(self):
        """Connect to XiaoZhi server."""
        from core.services.protocols.websocket_protocol import WebsocketProtocol

        self.protocol = WebsocketProtocol()

        # Wire up callbacks directly
        self.protocol.on_network_error = self._on_network_error
        self.protocol.on_incoming_audio = self._on_incoming_audio
        self.protocol.on_incoming_json = self._on_incoming_json
        self.protocol.on_audio_channel_opened = self._on_audio_channel_opened
        self.protocol.on_audio_channel_closed = self._on_audio_channel_closed

        return await self.protocol.connect()

    async def disconnect(self):
        """Disconnect from XiaoZhi server."""
        if self.protocol:
            await self.protocol.close_audio_channel()

    def is_connected(self) -> bool:
        """Check if connected to XiaoZhi server."""
        return self.protocol is not None and self.protocol.is_audio_channel_opened()

    # Audio initialization

    def init_audio(self):
        """Initialize audio codec and start audio input trigger."""
        try:
            from core.services.audio.codec import AudioCodec

            self._audio_codec = AudioCodec()
        except Exception as e:
            logger.warning(f"[XiaoZhi] Failed to initialize audio: {e}")
            return

        threading.Thread(target=self._audio_input_trigger, daemon=True).start()

    # Protocol callbacks

    def _on_network_error(self, message):
        """Handle network error."""
        self.set_device_state(DeviceState.IDLE)
        if self.device_state != DeviceState.CONNECTING:
            self.set_device_state(DeviceState.IDLE)
            asyncio.run_coroutine_threadsafe(
                self.disconnect(), self._app.loop
            )

    def _on_incoming_audio(self, data):
        """Handle incoming audio."""
        if self.device_state == DeviceState.SPEAKING:
            self._audio_codec.write_audio(data)

    def _on_incoming_json(self, json_data):
        """Handle incoming JSON message."""
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
        except Exception as exc:
            logger.error(
                f"[XiaoZhi] Failed to handle incoming json: {type(exc).__name__}: {exc}"
            )

    def _on_audio_channel_opened(self):
        """Handle audio channel opened."""
        self.set_device_state(DeviceState.IDLE)

    def _on_audio_channel_closed(self):
        """Handle audio channel closed."""
        self.set_device_state(DeviceState.IDLE)
        if self._audio_codec:
            self._audio_codec.stop_streams()

    # JSON message handlers

    def _handle_tts_message(self, data):
        """Handle TTS message."""
        state = data.get("state", "")
        if state == "start":
            self._app.schedule(lambda: self._handle_tts_start())
        elif state == "stop":
            self._app.schedule(lambda: self._handle_tts_stop())
        elif state == "sentence_start":
            text = data.get("text", "")
            if text:
                logger.ai_response(text, module="XiaoZhi")
                self._app.schedule(lambda: self._app.set_chat_message("assistant", text))

    def _handle_tts_start(self):
        """Handle TTS start."""
        if self.device_state in [DeviceState.IDLE, DeviceState.LISTENING]:
            self.set_device_state(DeviceState.SPEAKING)

    def _handle_tts_stop(self):
        """Handle TTS stop."""
        if self._tts_stop_future and not self._tts_stop_future.done() and self._session_loop:
            self._session_loop.call_soon_threadsafe(
                self._tts_stop_future.set_result, True
            )

    def _handle_stt_message(self, data):
        """Handle STT message."""
        text = data.get("text", "")
        if text:
            logger.user_speech(text, module="XiaoZhi")
            self._app.schedule(lambda: self._app.set_chat_message("user", text))

    def _handle_llm_message(self, data):
        """Handle LLM message."""
        text = data.get("text", "")
        if text:
            logger.ai_response(text, module="XiaoZhi")
            self._app.schedule(lambda: self._app.set_chat_message("assistant", text))
        emotion = data.get("emotion", "")
        if emotion:
            self._app.schedule(lambda: self._app.set_emotion(emotion))

    # Audio input handling

    def handle_input_audio(self):
        """Handle audio input - read and send to server."""
        if self.device_state != DeviceState.LISTENING:
            return

        encoded_data = self._audio_codec.read_audio()
        if encoded_data and self.is_connected():
            asyncio.run_coroutine_threadsafe(
                self.send_audio(encoded_data), self._app.loop
            )

    def _audio_input_trigger(self):
        """Trigger audio input event periodically."""
        while self._app and self._app.running:
            if self._audio_codec and self._audio_codec.input_stream.is_active():
                from core.services.protocols.typing import EventType
                self._app.events[EventType.AUDIO_INPUT_READY_EVENT].set()
            import time
            time.sleep(0.01)

    # User actions

    def start_listening(self):
        """Start listening."""
        self._app.schedule(self._start_listening_impl)

    def _start_listening_impl(self):
        """Start listening implementation."""
        self.set_device_state(DeviceState.IDLE)
        asyncio.run_coroutine_threadsafe(
            self.send_abort_speaking(AbortReason.ABORT),
            self._app.loop,
        )
        asyncio.run_coroutine_threadsafe(
            self.send_start_listening(ListeningMode.MANUAL), self._app.loop
        )
        self.set_device_state(DeviceState.LISTENING)

    def stop_listening(self):
        """Stop listening."""
        self._app.schedule(self._stop_listening_impl)

    def _stop_listening_impl(self):
        """Stop listening implementation."""
        asyncio.run_coroutine_threadsafe(
            self.send_stop_listening(), self._app.loop
        )
        self.set_device_state(DeviceState.IDLE)

    def abort_speaking(self, reason):
        """Abort speaking."""
        self.set_device_state(DeviceState.IDLE)
        asyncio.run_coroutine_threadsafe(
            self.send_abort_speaking(AbortReason.ABORT),
            self._app.loop,
        )

    # Send methods

    async def send_audio(self, frames):
        """Send audio frames."""
        if self.protocol:
            await self.protocol.send_audio(frames)

    async def send_text(self, text: str):
        """Send text message."""
        if self.protocol and self.protocol.is_audio_channel_opened():
            try:
                message = {
                    "type": "listen",
                    "mode": "manual",
                    "state": "detect",
                    "text": text,
                }
                await self.protocol.send_text(json.dumps(message))
                logger.info(f"[XiaoZhi] Text sent: {text}")
            except Exception as e:
                logger.error(f"[XiaoZhi] Failed to send text: {e}")
        else:
            logger.warning("[XiaoZhi] Not connected, cannot send text")

    async def send_start_listening(self, mode):
        """Send start listening command."""
        if self.protocol:
            await self.protocol.send_start_listening(mode)

    async def send_stop_listening(self):
        """Send stop listening command."""
        if self.protocol:
            await self.protocol.send_stop_listening()

    async def send_abort_speaking(self, reason):
        """Send abort speaking command."""
        if self.protocol:
            await self.protocol.send_abort_speaking(reason)

    # Wakeup session

    async def start_wakeup_session(self):
        """Start a VAD-driven wakeup session: notify → listen → silence → stop."""
        if not self.protocol:
            logger.warning("XiaoZhi is not ready, skip wakeup session", module="XiaoZhi")
            return

        self._session_loop = asyncio.get_running_loop()
        vad = get_vad()
        codec = get_audio_codec()
        speaker = get_speaker()

        self.set_device_state(DeviceState.IDLE)
        await self.send_abort_speaking(AbortReason.ABORT)

        if self._is_first_round:
            self._is_first_round = False
            await self._play_notify(speaker)

        # Wait for speech
        vad.resume("speech")
        result = await self._wait_vad_event(
            timeout=self.config.get_app_config("wakeup.timeout", 20)
        )
        if result is None:
            self.set_device_state(DeviceState.IDLE)
            logger.info("Wakeup timeout, exit listening", module="XiaoZhi")
            after_wakeup = self.config.get_app_config("wakeup.after_wakeup")
            if after_wakeup and speaker:
                await after_wakeup(speaker)
            return

        event_type, event_data = result
        if event_type != "speech":
            logger.debug(f"Expected speech, got {event_type}", module="XiaoZhi")
            return

        # Speech detected — send buffered audio and start listening
        logger.debug(
            f"VAD detected speech, buffer size: {len(event_data or b'')}",
            module="XiaoZhi",
        )
        set_speech_frames(event_data)
        if codec:
            codec.input_stream.start_stream()
        await self.send_start_listening(ListeningMode.MANUAL)
        self.set_device_state(DeviceState.LISTENING)

        # Wait for silence
        vad.resume("silence")
        result = await self._wait_vad_event()
        if result:
            event_type, _ = result
            logger.debug(f"VAD detected silence, stop listening", module="XiaoZhi")

        # Prepare TTS stop future before sending stop_listening,
        # so we don't miss the tts_stop event.
        self._tts_stop_future = self._session_loop.create_future()

        await self.send_stop_listening()
        self.set_device_state(DeviceState.IDLE)

        # Wait for TTS to finish, then start next round
        tts_finished = await self._wait_tts_stop(timeout=30)
        if tts_finished:
            await self.start_wakeup_session()

    async def _wait_vad_event(self, timeout=None):
        """Wait for a VAD event (speech/silence)."""
        from core.wakeup_session import EventManager

        self._vad_future = self._session_loop.create_future()
        original_on_speech = EventManager.on_speech
        original_on_silence = EventManager.on_silence

        def _on_speech(speech_buffer):
            if self._vad_future and not self._vad_future.done():
                self._session_loop.call_soon_threadsafe(
                    self._vad_future.set_result, ("speech", speech_buffer)
                )

        def _on_silence():
            if self._vad_future and not self._vad_future.done():
                self._session_loop.call_soon_threadsafe(
                    self._vad_future.set_result, ("silence", None)
                )

        EventManager.on_speech = _on_speech
        EventManager.on_silence = _on_silence

        try:
            return await asyncio.wait_for(self._vad_future, timeout=timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            EventManager.on_speech = original_on_speech
            EventManager.on_silence = original_on_silence
            self._vad_future = None

    async def _wait_tts_stop(self, timeout=None):
        """Wait for TTS playback to finish."""
        if not self._tts_stop_future:
            self._tts_stop_future = self._session_loop.create_future()
        try:
            return await asyncio.wait_for(self._tts_stop_future, timeout=timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self._tts_stop_future = None

    async def _play_notify(self, speaker):
        """Play the listening-ready notification sound."""
        if not _NOTIFY_PCM or not speaker:
            return
        try:
            await speaker.play(buffer=_NOTIFY_PCM)
            duration = len(_NOTIFY_PCM) / (24000 * 2)
            await asyncio.sleep(duration)
        except Exception as exc:
            logger.debug(f"Notify sound error: {exc}", module="XiaoZhi")

    # Shutdown

    def shutdown(self):
        """Shutdown XiaoZhi - close audio and disconnect."""
        if self._audio_codec:
            self._audio_codec.close()

        if self._app and self._app.loop:
            asyncio.run_coroutine_threadsafe(
                self.disconnect(), self._app.loop
            )
