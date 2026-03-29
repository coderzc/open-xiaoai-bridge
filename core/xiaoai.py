import asyncio
import threading
import time

import numpy as np
import open_xiaoai_server

from core.ref import get_speaker, set_xiaoai
from core.services.audio.stream import GlobalStream
from core.services.speaker import SpeakerManager
from core.wakeup_session import EventManager
from core.xiaoai_conversation import XiaoAIConversationController
from core.utils.base import json_decode
from core.utils.config import ConfigManager
from core.utils.logger import logger

ASCII_BANNER = """
▄▖      ▖▖▘    ▄▖▄▖
▌▌▛▌█▌▛▌▚▘▌▀▌▛▌▌▌▐ 
▙▌▙▌▙▖▌▌▌▌▌█▌▙▌▛▌▟▖
  ▌                
                                                                                                                
"""


class XiaoAI:
    speaker = SpeakerManager()
    async_loop: asyncio.AbstractEventLoop = None
    config_manager = ConfigManager.instance()
    conversation = XiaoAIConversationController()
    _input_gain_enabled = False
    _input_gain = 1.0
    _async_loop_ready = threading.Event()
    _external_wakeup_keywords: set[str] = set()
    _suppressed_dialog_ids: set[str] = set()
    _suppressed_dialog_last_attempt: dict[str, float] = {}
    _MAX_SUPPRESSED_DIALOGS = 100
    _SUPPRESS_RETRY_INTERVAL = 0.35

    @classmethod
    def refresh_runtime_config(cls, *_args):
        """从配置中心同步运行时参数。"""
        cls.conversation.apply_runtime_config(
            cls.config_manager.get_app_config("xiaoai", {})
        )
        gain = cls.config_manager.get_app_config("audio_input.gain", 1.0)
        try:
            gain = float(gain)
        except (TypeError, ValueError):
            gain = 1.0
        cls._input_gain = max(1.0, min(gain, 8.0))
        cls._input_gain_enabled = cls._input_gain > 1.0
        wakeup_keywords = cls.config_manager.get_app_config("wakeup.keywords", [])
        cls._external_wakeup_keywords = {
            cls._normalize_text(keyword)
            for keyword in wakeup_keywords
            if isinstance(keyword, str) and cls._normalize_text(keyword)
        }

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
        return "".join(text.strip().lower().split())

    @classmethod
    def _is_external_wakeup_text(cls, text: str) -> bool:
        normalized = cls._normalize_text(text)
        return bool(normalized) and normalized in cls._external_wakeup_keywords

    @classmethod
    async def _suppress_dialog(cls, dialog_id: str, reason: str):
        if not dialog_id:
            return

        # Prevent unbounded growth: if too many stale dialog_ids accumulated
        # (missed Dialog.Finish events), clear them all before adding new one
        if len(cls._suppressed_dialog_ids) >= cls._MAX_SUPPRESSED_DIALOGS:
            logger.debug(
                f"[XiaoAI] Clearing {len(cls._suppressed_dialog_ids)} stale suppressed dialog_ids"
            )
            cls._suppressed_dialog_ids.clear()
            cls._suppressed_dialog_last_attempt.clear()

        is_new_dialog = dialog_id not in cls._suppressed_dialog_ids
        cls._suppressed_dialog_ids.add(dialog_id)

        if is_new_dialog:
            logger.info(
                f"[XiaoAI] 🛑 停止小爱当前对话: {reason}"
            )

        now = time.monotonic()
        last_attempt = cls._suppressed_dialog_last_attempt.get(dialog_id, 0.0)
        if not is_new_dialog and (now - last_attempt) < cls._SUPPRESS_RETRY_INTERVAL:
            return

        cls._suppressed_dialog_last_attempt[dialog_id] = now
        try:
            await cls.speaker.run_shell(
                "killall tts_play.sh miplayer 2>/dev/null; mphelper pause"
            )
            if is_new_dialog:
                await cls.speaker.wake_up(awake=False)
        except Exception as exc:
            logger.debug(
                f"[XiaoAI] Failed to pause suppressed dialog {dialog_id}: {exc}"
            )

    @classmethod
    def on_input_data(cls, data: bytes):
        audio_array = np.frombuffer(data, dtype=np.int16)
        if cls._input_gain_enabled and audio_array.size > 0:
            boosted = audio_array.astype(np.float32) * cls._input_gain
            audio_array = np.clip(boosted, -32768, 32767).astype(np.int16)
        GlobalStream.input(audio_array.tobytes())

    @classmethod
    def on_output_data(cls, data: bytes):
        async def on_output_data_async(data: bytes):
            return await open_xiaoai_server.on_output_data(data)

        asyncio.run_coroutine_threadsafe(
            on_output_data_async(data),
            cls.async_loop,
        )

    @classmethod
    async def run_shell(cls, script: str, timeout: float = 10 * 1000):
        return await open_xiaoai_server.run_shell(script, timeout)

    @classmethod
    async def on_event(cls, event: str):
        event_json = json_decode(event) or {}
        if not isinstance(event_json, dict):
            logger.debug("[XiaoAI] 忽略非字典事件负载")
            return

        event_data = event_json.get("data", {})
        event_type = event_json.get("event")

        if not event_json.get("event"):
            return

        # 记录所有事件用于调试监听退出
        logger.debug(f"[XiaoAI] 📡 收到事件: {event_type} | 数据: {event_data}")

        if event_type == "instruction":
            if not isinstance(event_data, dict):
                logger.debug(
                    f"[XiaoAI] 忽略非字典 instruction 数据: {event_data}"
                )
                return

            raw_line = event_data.get("NewLine")
            if not raw_line:
                return

            line = json_decode(raw_line) if isinstance(raw_line, str) else raw_line
            if not isinstance(line, dict):
                logger.debug(f"[XiaoAI] 忽略无法解析的指令行: {raw_line}")
                return

            header = line.get("header", {}) if isinstance(line.get("header"), dict) else {}
            dialog_id = header.get("dialog_id", "")
            namespace = header.get("namespace")
            header_name = header.get("name")

            if dialog_id and dialog_id in cls._suppressed_dialog_ids:
                if namespace in {"Nlp", "SpeechSynthesizer", "AudioPlayer"}:
                    await cls._suppress_dialog(
                        dialog_id,
                        f"{namespace}.{header_name}",
                    )
                    return
                if namespace == "Dialog" and header_name == "Finish":
                    cls._suppressed_dialog_ids.discard(dialog_id)
                    cls._suppressed_dialog_last_attempt.pop(dialog_id, None)
                    logger.debug(
                        f"[XiaoAI] Cleared suppressed dialog: {dialog_id}"
                    )
                    return

            if (
                line
                and isinstance(line.get("header"), dict)
                and namespace == "SpeechRecognizer"
            ):
                if header_name == "RecognizeResult":
                    payload = line.get("payload", {})
                    if not isinstance(payload, dict):
                        logger.debug(
                            f"[XiaoAI] 忽略非字典 RecognizeResult payload: {payload}"
                        )
                        return

                    results = payload.get("results") or []
                    first_result = results[0] if results else {}
                    if isinstance(first_result, dict):
                        text = first_result.get("text") or ""
                    elif isinstance(first_result, str):
                        text = first_result
                    else:
                        text = ""
                    is_final = payload.get("is_final")
                    is_vad_begin = payload.get("is_vad_begin")

                    if EventManager.consume_openclaw_xiaoai_asr_result(
                        dialog_id=dialog_id,
                        text=text,
                        is_final=is_final,
                        is_vad_begin=is_vad_begin,
                    ):
                        if dialog_id and text and is_final:
                            await cls._suppress_dialog(
                                dialog_id,
                                f"OpenClaw 接管原生 ASR",
                            )
                        return
                    
                    # 只有明确的 is_vad_begin=False 且没有文本时才触发唤醒
                    # 避免重复触发
                    if not text and is_vad_begin is False:
                        logger.wakeup("小爱同学", module="XiaoAI")
                        cls.conversation.reset_retries()
                        EventManager.on_interrupt()
                    elif text and is_final and cls._is_external_wakeup_text(text):
                        await cls._suppress_dialog(
                            dialog_id,
                            f"外部唤醒词接管: {text}",
                        )
                        return
                    elif text and is_final:
                        logger.info(f"[XiaoAI] 🔥 收到指令: {text}")
                        cls.conversation.reset_retries()
                        await cls.conversation.handle_text_command(
                            text,
                            get_speaker(),
                        )
                        await EventManager.wakeup(text, "xiaoai")
                    elif is_final and not text:
                        logger.debug("[XiaoAI] 🛑 小爱监听超时自动退出")
                        await cls.conversation.handle_listening_timeout(
                            get_speaker()
                        )
            elif (
                line
                and isinstance(line.get("header"), dict)
                and namespace == "AudioPlayer"
            ):
                cls.conversation.handle_audio_player_instruction(header_name)
        elif event_type == "playing":
            if not isinstance(event_data, str):
                logger.debug(
                    f"[XiaoAI] 忽略非字符串 playing 数据: {event_data}"
                )
                return

            playing_status = event_data.lower()
            
            get_speaker().status = playing_status
            await cls.conversation.handle_playing_status(
                playing_status,
                get_speaker(),
            )
        
        else:
            # 记录未处理的事件类型，可能包含监听退出信息
            logger.debug(f"[XiaoAI] ❓ 未处理的事件类型: {event_type} | 完整数据: {event_json}")

    @classmethod
    def __init_background_event_loop(cls):
        def run_event_loop():
            cls.async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(cls.async_loop)
            cls._async_loop_ready.set()
            cls.async_loop.run_forever()

        thread = threading.Thread(target=run_event_loop, daemon=True)
        thread.start()
        cls._async_loop_ready.wait(timeout=2)

    @classmethod
    def __on_event(cls, event: str):
        future = asyncio.run_coroutine_threadsafe(
            cls.on_event(event),
            cls.async_loop,
        )

        def _log_result(done_future):
            try:
                done_future.result()
            except Exception as exc:
                logger.error(
                    f"[XiaoAI] Event handler failed: {type(exc).__name__}: {exc}"
                )

        future.add_done_callback(_log_result)

    @classmethod
    async def init_xiaoai(cls):
        cls.refresh_runtime_config()
        cls.config_manager.add_reload_listener(cls.refresh_runtime_config)
        set_xiaoai(XiaoAI)
        GlobalStream.on_output_data = cls.on_output_data
        open_xiaoai_server.register_fn("on_input_data", cls.on_input_data)
        open_xiaoai_server.register_fn("on_event", cls.__on_event)
        cls.__init_background_event_loop()
        logger.info("[XiaoAI] 启动小爱音箱服务...")
        print(ASCII_BANNER)
        await open_xiaoai_server.start_server()

    @classmethod
    def stop_conversation(cls):
        cls.conversation.stop()
