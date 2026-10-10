import asyncio
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.speech_queue import SequentialSpeechQueue


class SequentialSpeechQueueRaceTest(unittest.TestCase):
    def test_final_removes_progress_that_has_not_started(self):
        async def scenario():
            queue = SequentialSpeechQueue()
            await queue.put_progress("正在查看天气", key="weather")
            removed = await queue.mark_final_started()
            rejected = await queue.put_progress(
                "还在处理",
                key="working",
            )
            await queue.put_final("最终答案")
            await queue.close()
            item = await queue.get()
            await queue.finish(item)
            return removed, rejected, item, await queue.get()

        removed, rejected, item, end = asyncio.run(scenario())
        self.assertEqual(1, removed)
        self.assertFalse(rejected)
        self.assertEqual(("final", "最终答案"), (item.kind, item.text))
        self.assertIsNone(end)

    def test_progress_already_playing_finishes_before_final(self):
        async def scenario():
            queue = SequentialSpeechQueue()
            progress_started = asyncio.Event()
            release_progress = asyncio.Event()
            active = 0
            max_active = 0
            played = []

            async def play(item):
                nonlocal active, max_active
                active += 1
                max_active = max(max_active, active)
                played.append(("start", item.kind))
                if item.kind == "progress":
                    progress_started.set()
                    await release_progress.wait()
                played.append(("end", item.kind))
                active -= 1

            async def worker():
                while True:
                    item = await queue.get()
                    if item is None:
                        return
                    try:
                        await play(item)
                    finally:
                        await queue.finish(item)

            task = asyncio.create_task(worker())
            await queue.put_progress("正在查看天气", key="weather")
            await progress_started.wait()
            await queue.put_final("最终答案")
            await asyncio.sleep(0)
            self.assertNotIn(("start", "final"), played)
            release_progress.set()
            await queue.close()
            await task
            return max_active, played

        max_active, played = asyncio.run(scenario())
        self.assertEqual(1, max_active)
        self.assertEqual(
            [
                ("start", "progress"),
                ("end", "progress"),
                ("start", "final"),
                ("end", "final"),
            ],
            played,
        )

    def test_progress_is_deduplicated_and_coalesced(self):
        async def scenario():
            queue = SequentialSpeechQueue()
            first = await queue.put_progress(
                "正在查资料",
                key="research",
            )
            duplicate = await queue.put_progress(
                "仍在查资料",
                key="research",
            )
            newer = await queue.put_progress(
                "正在整理",
                key="organizing",
            )
            await queue.close()
            item = await queue.get()
            await queue.finish(item)
            return first, duplicate, newer, item, await queue.get()

        first, duplicate, newer, item, end = asyncio.run(scenario())
        self.assertTrue(first)
        self.assertFalse(duplicate)
        self.assertTrue(newer)
        self.assertEqual("organizing", item.key)
        self.assertIsNone(end)

    def test_abort_discards_all_queued_items_and_rejects_new_work(self):
        async def scenario():
            queue = SequentialSpeechQueue()
            await queue.put_final("第一句")
            first = await queue.get()
            await queue.put_final("第二句")
            await queue.put_final("第三句")

            removed = await queue.abort()
            repeated = await queue.abort()
            rejected_final = await queue.put_final("第四句")
            rejected_progress = await queue.put_progress(
                "仍在处理",
                key="working",
            )
            await queue.finish(first)
            return (
                removed,
                repeated,
                rejected_final,
                rejected_progress,
                queue.aborted,
                await queue.get(),
            )

        result = asyncio.run(scenario())
        self.assertEqual((2, 0, False, False, True, None), result)

    def test_abort_drops_progress_before_worker_can_take_it(self):
        async def scenario():
            queue = SequentialSpeechQueue()
            await queue.put_progress("正在处理", key="working")
            removed = await queue.abort()
            return removed, await queue.get()

        self.assertEqual((1, None), asyncio.run(scenario()))


if __name__ == "__main__":
    unittest.main()
