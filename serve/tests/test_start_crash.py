"""An engine the system ends at the start (SIGILL, a bad memory access, the OOM killer) writes nothing to its
log. It must not be retried as if a second try could fix it, and the error names how the process ended."""
from __future__ import annotations

import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.engine import process  # noqa: E402
from serve.engine.errors import EngineStartError  # noqa: E402
from serve.engine.progress import engine_crashed, engine_exit_words  # noqa: E402


class StatusWords(unittest.TestCase):
    def test_words(self):
        self.assertEqual(engine_exit_words(None), "")
        self.assertEqual(engine_exit_words(2), "exit code 2")
        self.assertIn("illegal instruction", engine_exit_words(0xC000001D).lower())
        self.assertIn("DLL", engine_exit_words(0xC0000135))

    def test_crashed(self):
        self.assertTrue(engine_crashed(0xC000001D) and engine_crashed(0xC0000005))
        self.assertFalse(engine_crashed(0) or engine_crashed(1) or engine_crashed(2) or engine_crashed(None))
        if os.name != "nt":
            self.assertIn("signal 4 (SIGILL", engine_exit_words(-int(signal.SIGILL)))
            for sig in (signal.SIGILL, signal.SIGSEGV, signal.SIGBUS, signal.SIGFPE, signal.SIGKILL):
                self.assertTrue(engine_crashed(-int(sig)), sig)
            self.assertFalse(engine_crashed(-int(signal.SIGABRT)))   # a teardown abort stays a retry
            self.assertFalse(engine_crashed(-int(signal.SIGTERM)))


class StartCrash(unittest.TestCase):
    def engine_for(self, code: str, tmp: str, made: list):
        script = Path(tmp) / "fake_engine.py"
        script.write_text("#!/usr/bin/env python3\n" + code, encoding="utf-8")
        script.chmod(0o755)

        class Hooked(process.ProcessEngine):
            def __init__(self, *a, **kw):
                made.append(self)                       # visible even when the start itself raises
                super().__init__(*a, **kw)

        return Hooked(str(script), [], cwd=tmp, log=None)

    def test_a_crashed_start_names_the_signal_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            spawns, sleeps, made = [], [], []
            real_popen = process.lifeline.popen

            def fake_popen(cmd, **kw):
                spawns.append(cmd)
                return real_popen(cmd, **kw)

            with mock.patch.object(process.lifeline, "popen", fake_popen), \
                    mock.patch.object(process.time, "sleep", sleeps.append):
                with self.assertRaises(EngineStartError) as first:
                    self.engine_for("import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n", tmp, made)
            self.assertIn("killed by signal 11 (SIGSEGV", str(first.exception))
            eng = made[0]
            with mock.patch.object(process.lifeline, "popen", fake_popen), \
                    mock.patch.object(process.time, "sleep", sleeps.append):
                with self.assertRaises(EngineStartError) as again:
                    eng.restart(tries=3)
            self.assertIn("killed by signal 11", str(again.exception))
            self.assertEqual(len(spawns), 2)             # the start, and exactly one restart try
            self.assertEqual(sleeps, [])                 # no retry waits: the crash was not retried

    def test_a_plain_exit_is_still_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            sleeps, made = [], []
            with self.assertRaises(EngineStartError):
                self.engine_for("import sys\nsys.exit(1)\n", tmp, made)
            eng = made[0]
            with mock.patch.object(process.time, "sleep", sleeps.append):
                with self.assertRaises(EngineStartError):
                    eng.restart(tries=3)
            self.assertEqual(len(sleeps), 2)             # the two waits between the three tries

    def test_the_note_carries_the_cpu_features_after_an_illegal_instruction(self):
        from serve.engine.progress import start_exit_note
        if os.name == "nt":
            self.assertIn("illegal instruction", start_exit_note(0xC000001D))
        else:
            self.assertIn("illegal instruction", start_exit_note(-int(signal.SIGILL)).lower())
        self.assertEqual(start_exit_note(None), "")
        self.assertEqual(start_exit_note(3), "\nthe engine process ended: exit code 3")


if __name__ == "__main__":
    unittest.main()
