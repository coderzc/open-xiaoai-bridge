"""流式对话后端使用的单工作线程语音队列。"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Literal


SpeechKind = Literal["progress", "final"]


@dataclass(frozen=True)
class SpeechItem:
    """一条等待顺序 TTS 播放的内容。"""

    kind: SpeechKind
    text: str
    key: str = ""


class SequentialSpeechQueue:
    """串行播放 TTS，并原子丢弃已被最终回答淘汰的进度。

    已经开始的内容不会被打断。若最终文本先到，删除待播进度；若进度已
    开始，则让它播完，再播放最终回答。
    """

    def __init__(self):
        self._condition = asyncio.Condition()
        self._items: deque[SpeechItem] = deque()
        self._closed = False
        self._aborted = False
        self._final_started = False
        self._playing: SpeechItem | None = None

    @property
    def final_started(self) -> bool:
        return self._final_started

    @property
    def playing(self) -> SpeechItem | None:
        return self._playing

    @property
    def aborted(self) -> bool:
        return self._aborted

    async def mark_final_started(self) -> int:
        """阻止后续进度，并原子删除待播进度。"""

        async with self._condition:
            if self._aborted or self._final_started:
                return 0
            self._final_started = True
            kept = deque(
                item for item in self._items
                if item.kind != "progress"
            )
            removed = len(self._items) - len(kept)
            self._items = kept
            self._condition.notify_all()
            return removed

    async def put_progress(self, text: str, *, key: str) -> bool:
        """最终文本尚未到达时，加入一条不重复的进度。"""

        async with self._condition:
            if self._closed or self._aborted or self._final_started:
                return False
            if (
                self._playing
                and self._playing.kind == "progress"
                and self._playing.key == key
            ):
                return False
            if any(
                item.kind == "progress" and item.key == key
                for item in self._items
            ):
                return False
            self._items = deque(
                item for item in self._items
                if item.kind != "progress"
            )
            self._items.append(SpeechItem("progress", text, key))
            self._condition.notify()
            return True

    async def put_final(self, text: str) -> bool:
        """按到达顺序加入最终文本。"""

        normalized = text.strip()
        if not normalized:
            return False
        await self.mark_final_started()
        async with self._condition:
            if self._closed or self._aborted:
                return False
            self._items.append(SpeechItem("final", normalized))
            self._condition.notify()
            return True

    async def get(self) -> SpeechItem | None:
        """返回下一条内容；关闭且队列耗尽后返回 ``None``。"""

        async with self._condition:
            while not self._items and not self._closed and not self._aborted:
                await self._condition.wait()
            if self._aborted or not self._items:
                return None
            item = self._items.popleft()
            self._playing = item
            return item

    async def finish(self, item: SpeechItem):
        """将一条内容标记为播放结束。"""

        async with self._condition:
            if self._playing is item:
                self._playing = None
            self._condition.notify_all()

    async def drain(self):
        """正常结束：停止接收新内容，并让已有队列自然耗尽。"""

        async with self._condition:
            if self._aborted:
                return
            self._closed = True
            self._condition.notify_all()

    async def close(self):
        """兼容旧调用；语义等同于 :meth:`drain`。"""

        await self.drain()

    async def abort(self) -> int:
        """主动中断：原子停止接收并丢弃全部尚未播放的内容。"""

        async with self._condition:
            if self._aborted:
                return 0
            self._aborted = True
            self._closed = True
            removed = len(self._items)
            self._items.clear()
            self._condition.notify_all()
            return removed
