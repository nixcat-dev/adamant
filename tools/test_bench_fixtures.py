"""The decode fixtures are one prompt each, ending where the model starts answering.

    python3 -m unittest tools.test_bench_fixtures
"""
import unittest
from pathlib import Path

BENCH = Path(__file__).resolve().parent.parent / "data" / "bench"


def ids(path: Path) -> list[int]:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(lines) == 1, f"{path}: one prompt per file"
    return [int(x) for x in lines[0].split(",")]


class Fixtures(unittest.TestCase):
    def test_every_decode_fixture_is_one_prompt_of_token_ids(self):
        for d in [BENCH, *sorted(p for p in BENCH.iterdir() if p.is_dir())]:
            for name in ("conv", "struct", "prose"):
                with self.subTest(dir=d.name, fixture=name):
                    self.assertTrue(all(i >= 0 for i in ids(d / f"{name}.ids")))


if __name__ == "__main__":
    unittest.main()
