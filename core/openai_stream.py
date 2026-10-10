"""协议中立的 OpenAI-compatible 文本流辅助组件。"""

from __future__ import annotations

import codecs
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import inspect
import json
import re
import time
from typing import Any

import aiohttp

from core.utils.logger import logger


class OpenAIStreamError(RuntimeError):
    """流式请求失败或服务端返回了不支持的响应。"""


@dataclass(frozen=True)
class SSEEvent:
    """一个已经解码的 Server-Sent Event。"""

    event: str
    data: str


class SSEDecoder:
    """增量解码 SSE，不假设网络分块恰好落在事件边界。"""

    def __init__(self):
        self._buffer = ""

    def feed(self, text: str, *, final: bool = False) -> list[SSEEvent]:
        self._buffer += text.replace("\r\n", "\n").replace("\r", "\n")
        blocks = self._buffer.split("\n\n")
        if final:
            self._buffer = ""
        else:
            self._buffer = blocks.pop()

        events: list[SSEEvent] = []
        for block in blocks:
            event = "message"
            data_lines: list[str] = []
            for line in block.splitlines():
                if not line or line.startswith(":"):
                    continue
                field, separator, value = line.partition(":")
                if separator and value.startswith(" "):
                    value = value[1:]
                if field == "event":
                    event = value or "message"
                elif field == "data":
                    data_lines.append(value)
            if data_lines:
                events.append(
                    SSEEvent(event=event, data="\n".join(data_lines))
                )
        return events


class SentenceChunker:
    """将流式文本整理为有序且适合 TTS 的句段。"""

    _BOUNDARY_RE = re.compile(r"[。！？!?；;\n]")

    def __init__(self, min_chars: int = 24, max_chars: int = 160):
        self.min_chars = max(1, int(min_chars))
        self.max_chars = max(self.min_chars, int(max_chars))
        self._buffer = ""

    def feed(self, delta: str, *, final: bool = False) -> list[str]:
        self._buffer += delta
        chunks: list[str] = []

        while True:
            boundaries = list(self._BOUNDARY_RE.finditer(self._buffer))
            if not boundaries:
                break

            cut = None
            for match in boundaries:
                if match.end() >= self.min_chars:
                    cut = match.end()
                    break
            if cut is None:
                if final:
                    cut = boundaries[-1].end()
                else:
                    break

            chunk = self._buffer[:cut].strip()
            self._buffer = self._buffer[cut:].lstrip()
            chunks.extend(self._split_oversized(chunk))

        if final and self._buffer.strip():
            chunks.extend(self._split_oversized(self._buffer.strip()))
            self._buffer = ""
        return [chunk for chunk in chunks if chunk]

    def _split_oversized(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        return [
            text[index:index + self.max_chars].strip()
            for index in range(0, len(text), self.max_chars)
            if text[index:index + self.max_chars].strip()
        ]


def extract_openai_delta(data: str) -> tuple[str, str | None]:
    """从一条 OpenAI SSE 数据中提取文本增量和结束原因。"""

    body = json.loads(data)
    choices = body.get("choices") if isinstance(body, dict) else None
    if not choices or not isinstance(choices[0], dict):
        return "", None
    choice = choices[0]
    delta = choice.get("delta")
    content = delta.get("content") if isinstance(delta, dict) else ""
    if isinstance(content, list):
        content = "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return (
        content if isinstance(content, str) else "",
        choice.get("finish_reason"),
    )


async def _invoke_callback(callback, value):
    result = callback(value)
    if inspect.isawaitable(result):
        await result


async def stream_openai_chat_completion(
    manager,
    text: str,
    *,
    on_delta: Callable[[str], Awaitable[None] | None],
    on_event: Callable[[SSEEvent], Awaitable[None] | None] | None = None,
    log_name: str = "OpenAI Stream",
) -> str:
    """流式接收标准 OpenAI 文本，不解释具名事件的业务语义。"""

    if not manager._initialized:
        manager.initialize_from_config()
    if not manager._enabled:
        raise OpenAIStreamError(f"{log_name} backend is disabled")

    session_key = manager._session_key
    scope_key = (
        manager._conversation_scope_key(session_key)
        if hasattr(manager, "_conversation_scope_key")
        else session_key
    )
    history = manager._sessions.setdefault(scope_key, [])
    messages = manager._build_messages(history, text)
    payload: dict[str, Any] = {
        "model": manager._model,
        "messages": messages,
        "stream": True,
        **manager._extra_body,
    }
    if manager._temperature is not None:
        payload["temperature"] = manager._temperature
    if manager._max_tokens is not None:
        payload["max_tokens"] = manager._max_tokens

    started_at = time.monotonic()
    text_parts: list[str] = []
    finish_reason: str | None = None
    decoder = SSEDecoder()
    utf8_decoder = codecs.getincrementaldecoder("utf-8")()

    async def consume_event(event: SSEEvent):
        nonlocal finish_reason
        if event.event != "message":
            if on_event:
                await _invoke_callback(on_event, event)
            return
        if event.data == "[DONE]":
            return
        try:
            delta, event_finish_reason = extract_openai_delta(event.data)
        except Exception as exc:
            logger.debug(
                f"Ignoring malformed SSE data event: {exc}",
                module=log_name,
            )
            return
        if event_finish_reason:
            finish_reason = str(event_finish_reason)
        if delta:
            text_parts.append(delta)
            await _invoke_callback(on_delta, delta)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=manager._timeout)
    ) as session:
        async with session.post(
            manager._chat_completions_url(),
            json=payload,
            headers=manager._headers(),
        ) as response:
            if response.status >= 400:
                error_text = await response.text()
                raise OpenAIStreamError(
                    f"HTTP {response.status}: {error_text[:500]}"
                )
            content_type = response.headers.get(
                "Content-Type",
                "",
            ).lower()
            if "text/event-stream" not in content_type:
                raise OpenAIStreamError(
                    f"Streaming unsupported: Content-Type={content_type!r}"
                )
            manager._capture_response_headers(
                response.headers,
                session_key=scope_key,
            )

            async for raw_chunk in response.content.iter_any():
                decoded = utf8_decoder.decode(raw_chunk)
                for event in decoder.feed(decoded):
                    await consume_event(event)

            tail = utf8_decoder.decode(b"", final=True)
            for event in decoder.feed(tail, final=True):
                await consume_event(event)

    if finish_reason == "error":
        raise OpenAIStreamError(
            "Server ended the stream with finish_reason=error"
        )

    response_text = "".join(text_parts).strip()
    if not response_text:
        raise OpenAIStreamError(
            "Streaming response contained no final text"
        )

    manager._append_history(history, text, response_text)
    logger.ai_response(
        response_text,
        module=f"{log_name}({session_key})",
    )
    logger.info(
        f"Streaming final response completed: chars={len(response_text)}, "
        f"elapsed={time.monotonic() - started_at:.1f}s",
        module=log_name,
    )
    return response_text
