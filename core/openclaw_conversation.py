"""OpenClaw continuous conversation controller.

After a custom wake word (e.g. "你好龙虾") triggers wakeup, this module
drives either:
  - local VAD -> ASR -> OpenClaw -> TTS
  - XiaoAI native ASR -> OpenClaw -> TTS

The selected input path runs independently of the XiaoZhi session state
machine.

Key design decisions:
  - Uses per-session asyncio.Future objects so it never conflicts with the
    XiaoZhi wakeup session state machine.
  - Never calls abort_xiaoai() (which would break the FileMonitor).
  - TTS playback is blocking (awaited), so the next listening round
    only starts after the response has finished playing.
"""

import asyncio
import os

import open_xiaoai_server

from core.openclaw import OpenClawManager
from core.ref import get_speaker, get_vad
from core.services.audio.asr import ASRManager
from core.utils.config import ConfigManager

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
from core.utils.logger import logger


class OpenClawConversationController:
    """Manages multi-turn OpenClaw conversation."""

    LOCAL_ASR_INPUT = "local_asr"
    XIAOAI_ASR_INPUT = "xiaoai_asr"
    XIAOAI_ASR_TIMEOUT = "__timeout__"

    def __init__(self):
        self.config = ConfigManager.instance()
        self.active = False

        # Per-session asyncio.Future used to receive VAD events
        self._vad_future: asyncio.Future | None = None
        # Per-session asyncio.Future used to receive XiaoAI native ASR results
        self._xiaoai_asr_future: asyncio.Future | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # Playback token for the current TTS session
        self._playback_token: int | None = None

    # ---- config helpers ----

    def _cfg(self, key: str, default=None):
        return self.config.get_app_config(f"openclaw.{key}", default)

    def _conversation_asr_cfg(self, key: str, default=None):
        return self.config.get_app_config(f"asr.contexts.conversation.{key}", default)

    @property
    def exit_keywords(self) -> list[str]:
        return self._cfg("exit_keywords", ["退出", "停止", "再见"])

    @property
    def timeout(self) -> int:
        return int(self.config.get_app_config("wakeup.timeout", 20))

    @property
    def input_mode(self) -> str:
        mode = self._cfg("input_mode", self.LOCAL_ASR_INPUT)
        if not isinstance(mode, str):
            return self.LOCAL_ASR_INPUT
        normalized = mode.strip().lower()
        if normalized in {self.LOCAL_ASR_INPUT, self.XIAOAI_ASR_INPUT}:
            return normalized
        logger.warning(
            f"Unknown openclaw.input_mode={mode!r}, fallback to {self.LOCAL_ASR_INPUT}",
            module="OpenClaw Conv",
        )
        return self.LOCAL_ASR_INPUT

    def uses_xiaoai_asr(self) -> bool:
        return self.input_mode == self.XIAOAI_ASR_INPUT

    # ---- public API ----

    def is_active(self) -> bool:
        return self.active

    async def start(self):
        """Enter OpenClaw conversation mode."""
        if self.active:
            logger.warning("[OpenClaw Conv] Already active, ignoring start()")
            return
        self.active = True
        self._loop = asyncio.get_running_loop()

        logger.info("🎙️ 进入 OpenClaw 连续对话模式", module="OpenClaw Conv")

        try:
            await self._conversation_loop()
        except Exception as exc:
            logger.error(
                f"Conversation loop error: {type(exc).__name__}: {exc}",
                module="OpenClaw Conv",
            )
        finally:
            self.stop()

    def stop(self):
        """Exit conversation mode and clean up."""
        if not self.active:
            return
        self.active = False
        self._cancel_vad_future()
        self._cancel_xiaoai_asr_future()
        if self._playback_token is not None:
            open_xiaoai_server.stop_tts_playback(self._playback_token)
            self._playback_token = None
        if self.uses_xiaoai_asr():
            speaker = get_speaker()
            if speaker and self._loop:
                try:
                    asyncio.run_coroutine_threadsafe(
                        speaker.wake_up(awake=False),
                        self._loop,
                    )
                except Exception as exc:
                    logger.debug(
                        f"Failed to stop XiaoAI native listening: {exc}",
                        module="OpenClaw Conv",
                    )
        logger.info("👋 退出 OpenClaw 连续对话模式", module="OpenClaw Conv")

    # ---- conversation loop ----

    async def _conversation_loop(self):
        """Run VAD -> ASR -> OpenClaw -> TTS turns until exit."""

        # Mute mic → play notify → unmute.
        # _play_notify() blocks for ~740ms (the beep duration), during which
        # the mic is off and before_wakeup TTS echo naturally fades.
        # VAD.resume() resets all state (speech_frames, input_bytes),
        # so speech detection starts clean when listening begins.
        await self._stop_recording()
        logger.debug("Recording stopped", module="OpenClaw Conv")
        await self._play_notify()
        await self._start_recording()
        logger.debug("Ready to listen", module="OpenClaw Conv")

        while self.active:
            if self.uses_xiaoai_asr():
                result = await self._run_one_turn_with_xiaoai_asr()
            else:
                result = await self._run_one_turn_with_local_asr()
            if result in ("exit", "timeout"):
                if self.uses_xiaoai_asr():
                    await self._stop_xiaoai_native_listening()
                await self._call_after_wakeup()
                break
            elif result == "error":
                break

    async def _run_one_turn_with_local_asr(self) -> str:
        """Execute a single conversation turn.

        Returns:
            "continue" - turn completed, loop to next
            "exit"     - user said an exit keyword
            "timeout"  - no speech detected within timeout
            "error"    - unrecoverable error
        """
        vad = get_vad()
        if not vad:
            logger.error("VAD not available", module="OpenClaw Conv")
            return "error"

        # 1. Start listening for speech (recording is already active)
        speech_bytes = await self._wait_for_speech(vad)
        if speech_bytes is None:
            return "timeout"

        logger.debug(
            f"Got speech buffer: {len(speech_bytes)} bytes",
            module="OpenClaw Conv",
        )

        # 2. ASR: convert speech to text
        text = ASRManager.asr(speech_bytes, sample_rate=16000)
        if not text:
            logger.debug("ASR empty, retrying", module="OpenClaw Conv")
            return "continue"

        # 3. Check exit keywords
        for kw in self.exit_keywords:
            if kw in text:
                logger.info(f"Exit keyword: {kw}", module="OpenClaw Conv")
                return "exit"

        # 4. Send to OpenClaw and wait for response
        full_text = text
        if OpenClawManager._rule_prompt:
            full_text = text + "\n" + OpenClawManager._rule_prompt
        response = await OpenClawManager.send(full_text, wait_response=True)
        if response is None:
            logger.warning("No response from OpenClaw", module="OpenClaw Conv")
            speaker = get_speaker()
            if speaker:
                await speaker.play(text="抱歉，我没有收到回复")
            return "continue"

        # 5. Stop recording → TTS → Notify → Start recording → Wait for silence
        #    Mic is off during TTS and notify, so no echo is captured.
        #    _play_notify() blocks for ~740ms (the beep duration),
        #    enough for TTS echo to fade. After starting recording, we wait
        #    for silence to ensure any residual echo or buffered audio clears.
        #    VAD.resume() resets all state, so speech detection starts clean.
        await self._stop_recording()
        await self._play_tts(str(response))
        await self._play_notify()
        await self._start_recording()
        logger.debug("Recording started, waiting for silence...", module="OpenClaw Conv")
        await self._wait_for_silence(vad)
        logger.debug("Ready to listen", module="OpenClaw Conv")

        return "continue"

    async def _run_one_turn_with_xiaoai_asr(self) -> str:
        """Execute a single conversation turn using XiaoAI native ASR."""
        text = await self._wait_for_xiaoai_asr_text()
        if text is None:
            logger.debug("XiaoAI native ASR turn timed out", module="OpenClaw Conv")
            return "timeout"

        for kw in self.exit_keywords:
            if kw in text:
                logger.info(f"Exit keyword: {kw}", module="OpenClaw Conv")
                return "exit"

        full_text = text
        if OpenClawManager._rule_prompt:
            full_text = text + "\n" + OpenClawManager._rule_prompt
        response = await OpenClawManager.send(full_text, wait_response=True)
        if response is None:
            logger.warning("No response from OpenClaw", module="OpenClaw Conv")
            speaker = get_speaker()
            if speaker:
                await speaker.play(text="抱歉，我没有收到回复")
            return "continue"

        await self._stop_recording()
        await self._play_tts(str(response))
        await self._play_notify()
        await self._start_recording()
        logger.debug("Ready for next XiaoAI native ASR round", module="OpenClaw Conv")
        return "continue"

    # ---- VAD integration ----

    async def _wait_for_speech(self, vad) -> bytes | None:
        """Use VAD to detect speech and collect complete utterance.

        Follows the same two-step pattern as the XiaoZhi wakeup session:
          1. resume("speech") → wait for on_speech (voice detected)
          2. resume("silence") → keep recording → wait for on_silence (user stopped)

        The audio stream is tapped between step 1 and 2 to capture the
        full utterance that the VAD does not provide by itself.

        Returns:
            PCM bytes of captured speech, or None on timeout.
        """
        from core.wakeup_session import EventManager

        self._vad_future = self._loop.create_future()
        recording_frames: list[bytes] = []
        is_recording = False
        max_speech_ms = int(self._conversation_asr_cfg("max_speech_ms", 30000) or 0)
        max_speech_bytes = int(self._conversation_asr_cfg("max_audio_bytes", 0) or 0)

        original_on_speech = EventManager.on_speech
        original_on_silence = EventManager.on_silence

        def _on_speech_hook(speech_buffer: bytes):
            """Voice detected — save initial buffer, start recording, wait for silence."""
            nonlocal is_recording
            recording_frames.append(speech_buffer)
            is_recording = True
            logger.debug(f"VAD speech detected, buffer size: {len(speech_buffer)}", module="OpenClaw Conv")
            # Now wait for silence to know user stopped speaking
            vad.resume("silence")

        def _on_silence_hook():
            """Silence detected — stop recording and resolve."""
            nonlocal is_recording
            is_recording = False
            logger.debug("VAD detected silence, stop recording", module="OpenClaw Conv")
            if self._vad_future and not self._vad_future.done():
                self._loop.call_soon_threadsafe(
                    self._vad_future.set_result, b"".join(recording_frames)
                )

        # Tap into VAD's audio stream to record frames while waiting for silence
        _orig_handle_speech = vad._handle_speech_frame
        _orig_handle_silence = vad._handle_silence_frame

        def _recording_speech_frame(frames):
            nonlocal is_recording
            if is_recording:
                recording_frames.append(bytes(frames))
                total_bytes = sum(len(x) for x in recording_frames)
                duration_ms = total_bytes * 1000 / (16000 * 2)
                if (max_speech_bytes > 0 and total_bytes > max_speech_bytes) or (
                    max_speech_ms > 0 and duration_ms > max_speech_ms
                ):
                    is_recording = False
                    logger.info(
                        f"[OpenClaw Conv] 语音片段超上限，直接丢弃: duration={duration_ms:.0f}ms, bytes={total_bytes}",
                        module="OpenClaw Conv",
                    )
                    recording_frames.clear()
                    vad.pause()
                    if self._vad_future and not self._vad_future.done():
                        self._loop.call_soon_threadsafe(self._vad_future.set_result, None)
                    return
            _orig_handle_speech(frames)

        def _recording_silence_frame(frames):
            nonlocal is_recording
            if is_recording:
                recording_frames.append(bytes(frames))
                total_bytes = sum(len(x) for x in recording_frames)
                duration_ms = total_bytes * 1000 / (16000 * 2)
                if (max_speech_bytes > 0 and total_bytes > max_speech_bytes) or (
                    max_speech_ms > 0 and duration_ms > max_speech_ms
                ):
                    is_recording = False
                    logger.info(
                        f"[OpenClaw Conv] 语音片段超上限，直接丢弃: duration={duration_ms:.0f}ms, bytes={total_bytes}",
                        module="OpenClaw Conv",
                    )
                    recording_frames.clear()
                    vad.pause()
                    if self._vad_future and not self._vad_future.done():
                        self._loop.call_soon_threadsafe(self._vad_future.set_result, None)
                    return
            _orig_handle_silence(frames)

        EventManager.on_speech = _on_speech_hook
        EventManager.on_silence = _on_silence_hook
        vad._handle_speech_frame = _recording_speech_frame
        vad._handle_silence_frame = _recording_silence_frame

        try:
            vad.resume("speech")
            result = await asyncio.wait_for(self._vad_future, timeout=self.timeout)
            return result

        except asyncio.TimeoutError:
            logger.debug("VAD timeout, no speech detected", module="OpenClaw Conv")
            vad.pause()
            return None

        finally:
            EventManager.on_speech = original_on_speech
            EventManager.on_silence = original_on_silence
            vad._handle_speech_frame = _orig_handle_speech
            vad._handle_silence_frame = _orig_handle_silence
            self._vad_future = None

    def _cancel_vad_future(self):
        """Cancel any pending VAD future."""
        if self._vad_future and not self._vad_future.done():
            self._loop.call_soon_threadsafe(self._vad_future.cancel)
        self._vad_future = None

    async def _wait_for_xiaoai_asr_text(self) -> str | None:
        """Wake XiaoAI and wait for a final native ASR result."""
        speaker = get_speaker()
        if not speaker:
            logger.error("Speaker not available", module="OpenClaw Conv")
            return None

        deadline = self._loop.time() + self.timeout
        while self.active:
            remaining = deadline - self._loop.time()
            if remaining <= 0:
                logger.info("XiaoAI native ASR hit outer wait timeout", module="OpenClaw Conv")
                return None

            self._xiaoai_asr_future = self._loop.create_future()
            try:
                wait_seconds = max(0.1, remaining)
                logger.debug(
                    f"Triggering XiaoAI native ASR, waiting up to {wait_seconds:.1f}s",
                    module="OpenClaw Conv",
                )
                await speaker.wake_up(awake=True, silent=True)
                result = await asyncio.wait_for(
                    self._xiaoai_asr_future,
                    timeout=wait_seconds,
                )
                if result == self.XIAOAI_ASR_TIMEOUT:
                    logger.debug(
                        "XiaoAI native ASR ended without speech (native timeout), retrying",
                        module="OpenClaw Conv",
                    )
                    continue
                return result
            except asyncio.TimeoutError:
                logger.info("XiaoAI native ASR hit outer wait timeout", module="OpenClaw Conv")
                return None
            finally:
                self._xiaoai_asr_future = None

        return None

    def consume_xiaoai_recognize_result(
        self,
        dialog_id: str,
        text: str,
        is_final,
        is_vad_begin,
    ) -> bool:
        """Consume XiaoAI native ASR events while OpenClaw is waiting."""
        if not (
            self.active
            and self.uses_xiaoai_asr()
            and self._xiaoai_asr_future
        ):
            return False

        normalized_text = text.strip() if isinstance(text, str) else ""
        if not is_final:
            logger.debug(
                f"Ignoring partial XiaoAI ASR result: {normalized_text}",
                module="OpenClaw Conv",
            )
            return True

        # Silent wakeup may emit an empty final marker before real speech starts.
        if not normalized_text and is_vad_begin is False:
            logger.debug("Ignoring XiaoAI wake marker for native ASR", module="OpenClaw Conv")
            return True

        if normalized_text:
            logger.debug(f"XiaoAI native ASR recognized: {normalized_text}", module="OpenClaw Conv")
            self._resolve_xiaoai_asr_future(normalized_text)
            return True

        logger.debug(
            "XiaoAI native ASR received empty final result",
            module="OpenClaw Conv",
        )
        self._resolve_xiaoai_asr_future(self.XIAOAI_ASR_TIMEOUT)
        return True

    def _resolve_xiaoai_asr_future(self, text: str):
        """Resolve the pending XiaoAI ASR future on the owner loop."""
        if self._xiaoai_asr_future and not self._xiaoai_asr_future.done():
            self._loop.call_soon_threadsafe(self._xiaoai_asr_future.set_result, text)

    def _cancel_xiaoai_asr_future(self):
        """Cancel any pending XiaoAI ASR future."""
        if self._xiaoai_asr_future and not self._xiaoai_asr_future.done():
            self._loop.call_soon_threadsafe(self._xiaoai_asr_future.cancel)
        self._xiaoai_asr_future = None

    async def _stop_xiaoai_native_listening(self):
        """Exit XiaoAI native listening before playing the goodbye prompt."""
        speaker = get_speaker()
        if not speaker:
            return
        try:
            await speaker.stop_device_audio()
            await speaker.wake_up(awake=False)
            await asyncio.sleep(0.15)
        except Exception as exc:
            logger.debug(
                f"Failed to stop XiaoAI native listening: {exc}",
                module="OpenClaw Conv",
            )

    async def _wait_for_silence(self, vad):
        """Wait until the environment is silent before starting to listen.

        Uses VAD silence detection to confirm the speaker has stopped playing.
        If speech is detected (e.g. TTS tail), keeps retrying silence detection.
        """
        from core.wakeup_session import EventManager

        future = self._loop.create_future()
        original_on_silence = EventManager.on_silence
        original_on_speech = EventManager.on_speech

        def _on_silence():
            if not future.done():
                self._loop.call_soon_threadsafe(future.set_result, True)

        def _on_speech(_speech_buffer: bytes):
            # Still hearing sound — switch back to silence detection
            vad.resume("silence")

        EventManager.on_silence = _on_silence
        EventManager.on_speech = _on_speech
        try:
            vad.resume("silence")
            await asyncio.wait_for(future, timeout=1)
        except asyncio.TimeoutError:
            vad.pause()
        finally:
            EventManager.on_speech = original_on_speech
            EventManager.on_silence = original_on_silence

    # ---- Recording control (physical mute/unmute via remote arecord) ----

    async def _stop_recording(self):
        """Kill the remote arecord process so the mic doesn't pick up TTS."""
        try:
            await open_xiaoai_server.stop_recording()
            logger.debug("Recording stopped", module="OpenClaw Conv")
        except Exception as exc:
            logger.debug(f"stop_recording error: {exc}", module="OpenClaw Conv")

    async def _start_recording(self):
        """Restart the remote arecord process to resume mic input."""
        try:
            await open_xiaoai_server.start_recording()
            logger.debug("Recording started", module="OpenClaw Conv")
        except Exception as exc:
            logger.debug(f"start_recording error: {exc}", module="OpenClaw Conv")

    # ---- TTS ----

    async def _play_tts(self, text: str):
        """Play text via Doubao TTS (blocks until playback finishes)."""
        self._playback_token = open_xiaoai_server.begin_playback_session()
        try:
            await OpenClawManager._play_response_with_tts(
                text,
                tts_speaker=OpenClawManager.get_tts_speaker_for_session_key(),
                playback_token=self._playback_token,
            )
        except Exception as exc:
            logger.error(
                f"TTS playback error: {exc}",
                module="OpenClaw Conv",
            )
            speaker = get_speaker()
            if speaker:
                await speaker.play(text=text)
        finally:
            self._playback_token = None

    async def _play_notify(self):
        """Play the listening-ready notification sound via PCM buffer."""
        if not _NOTIFY_PCM:
            return
        speaker = get_speaker()
        if speaker:
            try:
                await speaker.play(buffer=_NOTIFY_PCM)
                # Wait for playback to finish: PCM is int16 at 24000Hz
                duration = len(_NOTIFY_PCM) / (24000 * 2)
                await asyncio.sleep(duration)
            except Exception as exc:
                logger.debug(f"Notify sound error: {exc}", module="OpenClaw Conv")

    async def _call_after_wakeup(self):
        """Call the user-defined after_wakeup hook with source='openclaw'."""
        after_wakeup = self.config.get_app_config("wakeup.after_wakeup")
        if after_wakeup:
            speaker = get_speaker()
            if speaker:
                from core.openclaw import OpenClawManager
                await after_wakeup(speaker, source="openclaw", session_key=OpenClawManager._session_key)
