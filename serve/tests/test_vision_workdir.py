"""The vision work folder is made again when a temp cleaner removed it while the encoder was unloaded."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.engine.vision import Vision  # noqa: E402


class WorkDir(unittest.TestCase):
    def _bare(self, dir: Path) -> Vision:
        v = Vision.__new__(Vision)
        v.dir = dir
        v.lock = threading.Lock()
        v.cache: dict[str, tuple[Path, int, int]] = {}
        v.cache_bytes = 0
        v.pinned: dict[str, int] = {}
        return v

    def test_a_removed_folder_is_made_again_and_its_stale_cache_drops(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = self._bare(Path(tmp) / "vision-gone")
            v.cache = {"k": (v.dir / "k.sve", 4, 40)}
            v.cache_bytes = 40
            key = hashlib.sha256(b"data").hexdigest()[:32]

            def fake_enc(img: Path, out: Path) -> str:
                self.assertTrue(img.is_file())           # the folder exists before anything is written into it
                out.write_bytes(b"e" * 12)
                return "OK 7"

            v._enc = fake_enc
            pins: list[str] = []
            _, n, size = v._encode_one(b"data", pins)
            self.assertTrue(v.dir.is_dir())
            self.assertEqual(n, 7)
            self.assertEqual(pins, [key])
            self.assertEqual(set(v.cache), {key})        # the stale entry is gone, the fresh one took its place
            self.assertEqual(v.cache_bytes, size)

    def test_an_existing_folder_and_its_cache_are_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = self._bare(Path(tmp))
            v.cache = {"k": (Path(tmp, "k.sve"), 4, 4)}
            v.cache_bytes = 4
            v._ensure_dir()
            self.assertEqual(len(v.cache), 1)
            self.assertEqual(v.cache_bytes, 4)


if __name__ == "__main__":
    unittest.main()
