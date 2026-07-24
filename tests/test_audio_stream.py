import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.services.audio.stream import MyStream

FRAME = 512
FRAME_BYTES = FRAME * 2


class MyStreamOffsetTest(unittest.TestCase):
    def _new_stream(self) -> MyStream:
        return MyStream(
            rate=16000,
            channels=1,
            format=8,
            input=True,
            frames_per_buffer=FRAME,
            start=True,
        )

    def test_offset_read_returns_sequential_frames(self):
        stream = self._new_stream()
        stream.input(b"\x11" * (FRAME_BYTES * 3))

        self.assertEqual(FRAME_BYTES, len(stream.read(FRAME)))
        self.assertEqual(FRAME_BYTES, len(stream.read(FRAME)))
        self.assertEqual(2 * FRAME_BYTES, stream._read_offset)

    def test_clear_input_resets_read_cursor(self):
        stream = self._new_stream()
        stream.input(b"\x11" * (FRAME_BYTES * 3))
        stream.read(FRAME)
        stream.read(FRAME)
        self.assertNotEqual(0, stream._read_offset)

        stream.clear_input()

        self.assertEqual(0, stream._read_offset)
        self.assertEqual(0, len(stream.input_bytes))

    def test_fresh_audio_after_clear_is_read_from_start(self):
        """Regression: clearing the buffer must not make read() skip or drop
        freshly-arrived audio (the offset-based read bug from dbcbad9)."""
        stream = self._new_stream()
        stream.input(b"\x11" * (FRAME_BYTES * 3))
        stream.read(FRAME)
        stream.read(FRAME)  # advance _read_offset well past zero

        stream.clear_input()  # what VAD.resume() does between capture rounds

        fresh = b"\x22" * FRAME_BYTES
        stream.input(fresh)
        # Must return the fresh frame intact, not empty and not offset garbage.
        self.assertEqual(fresh, stream.read(FRAME))

    def test_stop_stream_also_resets_cursor(self):
        stream = self._new_stream()
        stream.input(b"\x11" * (FRAME_BYTES * 3))
        stream.read(FRAME)

        stream.stop_stream()

        self.assertEqual(0, stream._read_offset)
        self.assertEqual(0, len(stream.input_bytes))


if __name__ == "__main__":
    unittest.main()
