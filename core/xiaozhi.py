"""XiaoZhi protocol client - handles WebSocket connection to XiaoZhi server.

This module is responsible for:
- Connecting to XiaoZhi WebSocket server
- Sending/receiving audio and text messages
- Protocol-level message handling

The main application flow is managed by main.py.
"""

import asyncio
import json

from core.services.protocols.protocol import Protocol
from core.utils.config import ConfigManager
from core.utils.logger import logger


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

        # Callbacks (set by MainApp)
        self.on_network_error = None
        self.on_incoming_audio = None
        self.on_incoming_json = None
        self.on_audio_channel_opened = None
        self.on_audio_channel_closed = None

        # References (set by MainApp)
        self._app = None
        self._audio_codec = None
        self._display = None

    def set_app(self, app):
        """Set reference to main app."""
        self._app = app

    def set_audio_codec(self, codec):
        """Set audio codec reference."""
        self._audio_codec = codec

    def set_display(self, display):
        """Set display reference."""
        self._display = display

    async def connect(self):
        """Connect to XiaoZhi server."""
        from core.services.protocols.websocket_protocol import WebsocketProtocol

        self.protocol = WebsocketProtocol()

        # Wire up callbacks
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

    # Protocol callbacks (forward to app)
    def _on_network_error(self, message):
        """Handle network error."""
        if self.on_network_error:
            self.on_network_error(message)

    def _on_incoming_audio(self, data):
        """Handle incoming audio."""
        if self.on_incoming_audio:
            self.on_incoming_audio(data)

    def _on_incoming_json(self, data):
        """Handle incoming JSON."""
        if self.on_incoming_json:
            self.on_incoming_json(data)

    def _on_audio_channel_opened(self):
        """Handle audio channel opened."""
        if self.on_audio_channel_opened:
            self.on_audio_channel_opened()

    def _on_audio_channel_closed(self):
        """Handle audio channel closed."""
        if self.on_audio_channel_closed:
            self.on_audio_channel_closed()

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
