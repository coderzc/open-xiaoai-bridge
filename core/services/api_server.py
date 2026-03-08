"""
HTTP API Server for XiaoZhi
Provides endpoints to play text/audio remotely
"""

import asyncio
import json
import io
import numpy as np
import soundfile as sf
from aiohttp import web
from core.ref import get_speaker, get_xiaoai
from core.utils.logger import logger
from core.services.tts.doubao import DoubaoTTS


def decode_audio_to_pcm(audio_data: bytes, filename: str, target_sample_rate: int = 24000) -> bytes:
    """
    Decode audio file to PCM format (16-bit, target_sample_rate Hz, mono)
    Supports WAV, MP3, OGG, FLAC and other formats via soundfile

    Args:
        audio_data: Raw audio file bytes
        filename: Original filename (for format detection)
        target_sample_rate: Target sample rate in Hz (default 24000, can be 48000, 44100, etc.)
    """
    ext = filename.lower().split('.')[-1] if '.' in filename else ''

    try:
        # Use soundfile to read audio data
        audio_array, sample_rate = sf.read(io.BytesIO(audio_data), dtype='float32')

        logger.info(f"[APIServer] Audio: shape={audio_array.shape}, rate={sample_rate}, format={ext}")

        # Convert to mono if stereo
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample if needed (high-quality using scipy)
        if sample_rate != target_sample_rate:
            audio_array = resample_audio(audio_array, sample_rate, target_sample_rate)

        # Convert back to 16-bit signed PCM
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)

        return audio_array.tobytes()

    except Exception as e:
        logger.error(f"[APIServer] Failed to decode audio: {e}")
        # Fallback: return original data
        return audio_data


def resample_audio(audio_array: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    High-quality resampling using scipy.signal.resample_poly
    Uses polyphase filtering for better quality than linear interpolation
    """
    if src_rate == dst_rate:
        return audio_array

    try:
        from scipy import signal

        # Calculate greatest common divisor for efficient resampling
        gcd = np.gcd(src_rate, dst_rate)
        up = dst_rate // gcd
        down = src_rate // gcd

        # Use polyphase resampling (high quality)
        resampled = signal.resample_poly(audio_array, up, down, window=('kaiser', 5.0))

        logger.info(f"[APIServer] Resampled from {src_rate}Hz to {dst_rate}Hz using polyphase filter (up={up}, down={down})")
        return resampled

    except ImportError:
        logger.warning("[APIServer] scipy not available, falling back to linear interpolation")
        # Fallback to linear interpolation
        src_length = len(audio_array)
        dst_length = int(src_length * dst_rate / src_rate)
        indices = np.linspace(0, src_length - 1, dst_length)
        indices_low = np.floor(indices).astype(np.int32)
        indices_high = np.minimum(indices_low + 1, src_length - 1)
        fractions = indices - indices_low
        return audio_array[indices_low] * (1 - fractions) + audio_array[indices_high] * fractions


class APIServer:
    """HTTP API Server to control XiaoZhi speaker remotely"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self.host = host
        self.port = port
        self.app = web.Application()
        self.runner = None
        self.site = None
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes"""
        self.app.router.add_post("/api/play/text", self.handle_play_text)
        self.app.router.add_post("/api/play/url", self.handle_play_url)
        self.app.router.add_post("/api/play/file", self.handle_play_file)
        self.app.router.add_get("/api/status", self.handle_get_status)
        self.app.router.add_post("/api/wakeup", self.handle_wakeup)
        self.app.router.add_post("/api/interrupt", self.handle_stop)
        self.app.router.add_get("/api/health", self.handle_health)
        # TTS endpoints
        self.app.router.add_post("/api/tts/doubao", self.handle_tts_doubao)
        self.app.router.add_get("/api/tts/doubao_voices", self.handle_tts_voices)

    async def start(self):
        """Start the HTTP server"""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        logger.info(f"[APIServer] HTTP server started at http://{self.host}:{self.port}")

    async def stop(self):
        """Stop the HTTP server"""
        if self.runner:
            await self.runner.cleanup()
            logger.info("[APIServer] HTTP server stopped")

    # ============ Handlers ============

    async def handle_play_text(self, request: web.Request) -> web.Response:
        """
        POST /api/play/text
        Play text via TTS

        Request body:
            {
                "text": "你好",           # required
                "blocking": false,        # optional, default false
                "timeout": 60000          # optional, timeout in ms
            }
        """
        try:
            data = await request.json()
            text = data.get("text")

            if not text:
                return web.json_response(
                    {"success": False, "error": "Missing required field: text"},
                    status=400
                )

            blocking = data.get("blocking", False)
            timeout = data.get("timeout", 10 * 60 * 1000)

            speaker = get_speaker()
            if not speaker:
                return web.json_response(
                    {"success": False, "error": "Speaker not initialized"},
                    status=503
                )

            # Run in background to not block the response
            if blocking:
                result = await speaker.play(text=text, blocking=True, timeout=timeout)
                return web.json_response({"success": result})
            else:
                asyncio.create_task(speaker.play(text=text, blocking=False, timeout=timeout))
                return web.json_response({"success": True, "message": "Playing text in background"})

        except json.JSONDecodeError:
            return web.json_response(
                {"success": False, "error": "Invalid JSON"},
                status=400
            )
        except Exception as e:
            logger.error(f"[APIServer] Error playing text: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500
            )

    async def handle_play_url(self, request: web.Request) -> web.Response:
        """
        POST /api/play/url
        Play audio from URL

        Request body:
            {
                "url": "http://example.com/audio.mp3",  # required
                "blocking": false,                       # optional, default false
                "timeout": 60000                         # optional, timeout in ms
            }
        """
        try:
            data = await request.json()
            url = data.get("url")

            if not url:
                return web.json_response(
                    {"success": False, "error": "Missing required field: url"},
                    status=400
                )

            blocking = data.get("blocking", False)
            timeout = data.get("timeout", 10 * 60 * 1000)

            speaker = get_speaker()
            if not speaker:
                return web.json_response(
                    {"success": False, "error": "Speaker not initialized"},
                    status=503
                )

            if blocking:
                result = await speaker.play(url=url, blocking=True, timeout=timeout)
                return web.json_response({"success": result})
            else:
                asyncio.create_task(speaker.play(url=url, blocking=False, timeout=timeout))
                return web.json_response({"success": True, "message": "Playing URL in background"})

        except json.JSONDecodeError:
            return web.json_response(
                {"success": False, "error": "Invalid JSON"},
                status=400
            )
        except Exception as e:
            logger.error(f"[APIServer] Error playing URL: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500
            )

    async def handle_play_file(self, request: web.Request) -> web.Response:
        """
        POST /api/play/file
        Upload and play audio file directly via audio buffer

        Request: multipart/form-data
            - file: audio file (required, mp3/wav/opus etc.)

        Query params:
            - blocking: true/false (optional, default false)
            - sample_rate: target sample rate in Hz (optional, default 24000, can be 48000, 44100, etc.)

        Response:
            {
                "success": true,
                "message": "File played"
            }
        """
        try:
            # Parse query params
            blocking = request.query.get("blocking", "false").lower() == "true"
            sample_rate = int(request.query.get("sample_rate", "24000"))

            reader = await request.multipart()

            # Get the file field
            field = await reader.next()
            if not field or field.name != "file":
                return web.json_response(
                    {"success": False, "error": "Missing required field: file"},
                    status=400
                )

            # Check filename
            filename = field.filename
            if not filename:
                return web.json_response(
                    {"success": False, "error": "No filename provided"},
                    status=400
                )

            # Read file content into memory
            audio_data = bytearray()
            while True:
                chunk = await field.read_chunk(size=8192)
                if not chunk:
                    break
                audio_data.extend(chunk)

            logger.info(f"[APIServer] Received file: {filename}, size: {len(audio_data)} bytes, blocking={blocking}, sample_rate={sample_rate}")

            speaker = get_speaker()
            if not speaker:
                return web.json_response(
                    {"success": False, "error": "Speaker not initialized"},
                    status=503
                )

            # Decode audio file to PCM format with custom sample rate
            pcm_data = decode_audio_to_pcm(bytes(audio_data), filename, target_sample_rate=sample_rate)
            logger.info(f"[APIServer] Decoded to PCM: {len(pcm_data)} bytes at {sample_rate}Hz")

            # Play audio buffer directly (chunked to avoid WebSocket message size limit)
            async def play_audio():
                try:
                    # 分块发送，每块 1MB (避免超过 WebSocket 16MB 限制)
                    # Note: 实测 1MB 是稳定的大小，Base64 后约 1.3MB，安全余量充足
                    CHUNK_SIZE = 1 * 1024 * 1024  # 1MB
                    total_size = len(pcm_data)
                    offset = 0

                    logger.info(f"[APIServer] Playing file in chunks: {total_size} bytes, chunk_size={CHUNK_SIZE}")

                    while offset < total_size:
                        chunk = pcm_data[offset:offset + CHUNK_SIZE]
                        await speaker.play(buffer=chunk, blocking=False)
                        offset += len(chunk)
                        # 增加延迟，确保每块发送完成，避免队列累积
                        await asyncio.sleep(0.05)

                    logger.info(f"[APIServer] Finished playing: {filename}")
                    return True
                except Exception as e:
                    logger.error(f"[APIServer] Error playing file: {e}")
                    return False

            if blocking:
                # Wait for playback to complete
                success = await play_audio()
                return web.json_response({
                    "success": success,
                    "message": f"Finished playing: {filename}",
                    "filename": filename,
                    "size": len(audio_data),
                    "sample_rate": sample_rate
                })
            else:
                # Run in background
                asyncio.create_task(play_audio())
                return web.json_response({
                    "success": True,
                    "message": f"Playing file: {filename}",
                    "filename": filename,
                    "size": len(audio_data),
                    "sample_rate": sample_rate
                })

        except Exception as e:
            logger.error(f"[APIServer] Error handling file upload: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500
            )

    async def handle_get_status(self, request: web.Request) -> web.Response:
        """
        GET /api/status
        Get current speaker status
        """
        try:
            speaker = get_speaker()

            if not speaker:
                return web.json_response(
                    {"success": False, "error": "Speaker not initialized"},
                    status=503
                )

            status = await speaker.get_playing()

            return web.json_response({
                "success": True,
                "data": {
                    "status": status  # "playing", "paused", "idle"
                }
            })

        except Exception as e:
            logger.error(f"[APIServer] Error getting status: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500
            )

    async def handle_wakeup(self, request: web.Request) -> web.Response:
        """
        POST /api/wakeup
        Wake up the speaker

        Request body:
            {
                "silent": true   # optional, default true (silent wakeup)
            }
        """
        try:
            data = await request.json() if request.can_read_body else {}
            silent = data.get("silent", True)

            speaker = get_speaker()
            if not speaker:
                return web.json_response(
                    {"success": False, "error": "Speaker not initialized"},
                    status=503
                )

            result = await speaker.wake_up(awake=True, silent=silent)
            return web.json_response({"success": result})

        except Exception as e:
            logger.error(f"[APIServer] Error waking up: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500
            )

    async def handle_stop(self, request: web.Request) -> web.Response:
        """
        POST /api/interrupt
        Interrupt current playback
        """
        try:
            speaker = get_speaker()
            xiaoai = get_xiaoai()
            if not speaker:
                return web.json_response(
                    {"success": False, "error": "Speaker not initialized"},
                    status=503
                )

            # 打断原来的小爱同学
            await speaker.abort_xiaoai()
            # 停止连续对话
            if xiaoai:
                xiaoai.stop_conversation()
            # 等待 2 秒，让小爱 TTS 恢复可用
            import time
            time.sleep(2)

            return web.json_response({"success": True})

        except Exception as e:
            logger.error(f"[APIServer] Error interrupting: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500
            )

    async def handle_health(self, request: web.Request) -> web.Response:
        """
        GET /api/health
        Health check endpoint
        """
        return web.json_response({
            "success": True,
            "data": {
                "status": "healthy",
                "speaker_ready": get_speaker() is not None
            }
        })

    async def handle_tts_doubao(self, request: web.Request) -> web.Response:
        """
        POST /api/tts/doubao
        Synthesize text using Doubao (ByteDance Volcano) TTS and play it

        Request body:
            {
                "text": "你好",                    # required
                "app_id": "your_app_id",           # optional (uses config if not provided)
                "access_key": "your_access_key",   # optional (uses config if not provided)
                "resource_id": "your_resource_id", # optional (auto-detected based on speaker)
                "speaker": "zh_female_cancan_mars_bigtts",  # optional, default voice
                "speed": 1.0,                       # optional, 0.8-2.0
                "blocking": true,                   # optional, default false
                "emotion": "happy",                 # optional, emotion for multi-emotion speakers
                "context_texts": [                   # optional, only for 2.0 speakers (only first value effective)
                    "你可以说慢一点吗？",
                    "你可以用特别痛心的语气说话吗？",
                    "你能用骄傲的语气来说话吗？"
                ]
            }
        """
        try:
            data = await request.json()
            text = data.get("text")

            if not text:
                return web.json_response(
                    {"success": False, "error": "Missing required field: text"},
                    status=400
                )

            # Get credentials from request or config
            from config import APP_CONFIG
            tts_config = APP_CONFIG.get("tts", {}).get("doubao", {})

            app_id = data.get("app_id") or tts_config.get("app_id")
            access_key = data.get("access_key") or tts_config.get("access_key")
            # resource_id is now optional - will be auto-detected based on speaker
            resource_id = data.get("resource_id") or tts_config.get("resource_id")

            if not all([app_id, access_key]):
                return web.json_response(
                    {"success": False, "error": "Doubao TTS credentials not configured. Provide app_id and access_key in request or config.py"},
                    status=400
                )

            speaker_id = data.get("speaker") or tts_config.get("default_speaker") or "zh_female_shuangkuaisisi_moon_bigtts"
            speed = float(data.get("speed", 1.0))
            blocking = data.get("blocking", False)
            context_texts = data.get("context_texts")  # Only supported for 2.0 speakers
            emotion = data.get("emotion")  # Emotion parameter for multi-emotion speakers

            speaker = get_speaker()
            if not speaker:
                return web.json_response(
                    {"success": False, "error": "Speaker not initialized"},
                    status=503
                )

            # Create TTS instance (auto-detects resource_id if not provided)
            tts = DoubaoTTS(
                app_id=app_id,
                access_key=access_key,
                resource_id=resource_id,
                speaker=speaker_id,
            )
            logger.info(f"[APIServer] Doubao TTS: speaker={speaker_id}, resource_id={tts.resource_id}")

            # Synthesize in thread pool to not block
            try:
                loop = asyncio.get_event_loop()
                audio_data = await loop.run_in_executor(
                    None, lambda: tts.synthesize(text, speed=speed, context_texts=context_texts, emotion=emotion)
                )

                if not audio_data:
                    return web.json_response(
                        {"success": False, "error": "TTS synthesis returned empty audio"},
                        status=500
                    )
            except Exception as e:
                logger.error(f"[APIServer] TTS synthesis error: {e}")
                return web.json_response(
                    {"success": False, "error": f"TTS synthesis failed: {str(e)}"},
                    status=500
                )

            logger.info(f"[APIServer] Doubao TTS synthesized: {len(audio_data)} bytes, speaker={speaker_id}")

            # Play the synthesized audio (chunked only if large)
            async def play_tts_audio():
                try:
                    total_size = len(audio_data)
                    # 只有超过 2MB 才分块，小文件直接发送
                    CHUNK_THRESHOLD = 2 * 1024 * 1024  # 2MB
                    CHUNK_SIZE = 1 * 1024 * 1024  # 1MB (实测稳定的块大小)

                    if total_size <= CHUNK_THRESHOLD:
                        # 小文件直接发送
                        logger.info(f"[APIServer] Playing TTS audio directly: {total_size} bytes")
                        await speaker.play(buffer=audio_data, blocking=True)
                    else:
                        # 大文件分块发送
                        offset = 0
                        logger.info(f"[APIServer] Playing TTS audio in chunks: {total_size} bytes, chunk_size={CHUNK_SIZE}")
                        while offset < total_size:
                            chunk = audio_data[offset:offset + CHUNK_SIZE]
                            await speaker.play(buffer=chunk, blocking=False)
                            offset += len(chunk)
                            await asyncio.sleep(0.05)

                    logger.info(f"[APIServer] Finished playing TTS audio")
                    return True
                except Exception as e:
                    logger.error(f"[APIServer] Error playing TTS audio: {e}")
                    return False

            if blocking:
                success = await play_tts_audio()
                return web.json_response({
                    "success": success,
                    "message": f"TTS played: {text[:50]}..." if len(text) > 50 else f"TTS played: {text}",
                    "speaker": speaker_id,
                    "audio_size": len(audio_data)
                })
            else:
                asyncio.create_task(play_tts_audio())
                return web.json_response({
                    "success": True,
                    "message": f"TTS playing in background: {text[:50]}..." if len(text) > 50 else f"TTS playing: {text}",
                    "speaker": speaker_id,
                    "audio_size": len(audio_data)
                })

        except json.JSONDecodeError:
            return web.json_response(
                {"success": False, "error": "Invalid JSON"},
                status=400
            )
        except Exception as e:
            logger.error(f"[APIServer] Error in Doubao TTS: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500
            )

    async def handle_tts_voices(self, request: web.Request) -> web.Response:
        """
        GET /api/tts/voices
        Get available TTS voices for Doubao

        Query params:
            - version: "1.0", "2.0", or "all" (optional, default shows all)
        """
        try:
            from config import APP_CONFIG
            tts_config = APP_CONFIG.get("tts", {}).get("doubao", {})
            resource_id = tts_config.get("resource_id", "")

            # Get version from query param or auto-detect from resource_id
            version = request.query.get("version", "all")

            if version == "2.0":
                voices = DoubaoTTS.VOICES_2_0
            elif version == "1.0":
                voices = DoubaoTTS.VOICES_1_0
            else:
                voices = DoubaoTTS.list_voices()
                # Add version info for all voices
                return web.json_response({
                    "success": True,
                    "data": {
                        "provider": "doubao",
                        "resource_id": resource_id,
                        "versions": {
                            "1.0": {
                                "count": len(DoubaoTTS.VOICES_1_0),
                                "description": "豆包语音合成模型1.0",
                                "sample_voices": list(DoubaoTTS.VOICES_1_0.keys())[:5]
                            },
                            "2.0": {
                                "count": len(DoubaoTTS.VOICES_2_0),
                                "description": "豆包语音合成模型2.0 - 支持情感变化、指令遵循、ASMR",
                                "sample_voices": list(DoubaoTTS.VOICES_2_0.keys())[:5]
                            }
                        },
                        "total_voices": len(voices)
                    }
                })

            return web.json_response({
                "success": True,
                "data": {
                    "provider": "doubao",
                    "version": version,
                    "resource_id": resource_id,
                    "voices": voices,
                    "count": len(voices)
                }
            })
        except Exception as e:
            logger.error(f"[APIServer] Error getting voices: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500
            )
