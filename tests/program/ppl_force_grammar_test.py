#!/usr/bin/env python3
"""Check fixed-text PPL replay against grammar and relaxed-drafting controls."""
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tools"))
from ppl_gate import read_base
from model_file import ModelFile


def request(proc, prompt, count, keys):
    proc.stdin.write(f"GEN {count} {keys} {','.join(map(str, prompt))}\n")
    proc.stdin.flush()
    tokens = []
    for line in proc.stdout:
        if line.startswith("T "):
            tokens.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            return tokens
        elif line.startswith("ERR"):
            raise RuntimeError(line.strip())
    raise RuntimeError("engine exited during request")


def main():
    engine, model, base, *extra = sys.argv[1:]
    synthetic = base == "-"
    if synthetic:
        vocab = json.loads(ModelFile(model).read("tokenizer/vocab.json"))
        n_ctx, n_vocab = 16, len(vocab)
        ordinary = [vocab[c] for c in "abcdefghijklmnopqrstuvwxyz" if c in vocab]
        if len(ordinary) < 2:
            raise RuntimeError("the model tokenizer has no ordinary letter tokens")
        first = n_ctx // 2
        prompt = [ordinary[i % len(ordinary)] for i in range(first + 1)]
    else:
        n_ctx, _, _, chunks = read_base(base)
        first = n_ctx // 2
        prompt = chunks[0][:first + 1]
        expected = list(chunks[0][first + 1:])
    schema = {"type": "string", "enum": ["fixed grammar control"]}
    grammar = "grammar=" + json.dumps({"kind": "json_schema", "schema": schema}, separators=(",", ":")).replace(" ", "\\u0020")
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile(mode="w+") as log:
        rows = pathlib.Path(tmp) / "rows.tsv"
        base_path = pathlib.Path(tmp) / "synthetic.kld" if synthetic else pathlib.Path(base)
        proc = subprocess.Popen([engine, "--serve", "--model", model, "--grammar", "on", "--spec", "8",
                                 "--ppl-base", str(base_path), "--ppl-out", str(rows), *extra],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                text=True, bufsize=1)
        try:
            for line in proc.stdout:
                if line.startswith("READY"):
                    break
            else:
                log.seek(0)
                raise RuntimeError(f"engine exited before READY: {log.read()}")
            control = request(proc, prompt, 1, f"temperature=0 {grammar}")
            if not control:
                raise RuntimeError("grammar control produced no token")
            if synthetic:
                mismatch = next(token for token in ordinary if token != control[0])
                expected = [mismatch] + [ordinary[i % len(ordinary)] for i in range(1, n_ctx - first - 1)]
                tokens = prompt + expected
                with base_path.open("wb") as f:
                    f.write(b"_logits_")
                    f.write(struct.pack("<Iii", n_ctx, n_vocab, 1))
                    f.write(struct.pack(f"<{len(tokens)}i", *tokens))
            if control[0] == expected[0]:
                raise RuntimeError("grammar control did not conflict with the first reference token; use another base")
            replay = request(proc, prompt, len(expected), "ppl=0 ppl_t=3 ppl_score=0")
            if replay != expected:
                raise RuntimeError(f"score-off teacher-forcing differs: {replay!r} != {expected!r}")
            for keys in ("", "relax=0.01", grammar, f"relax=0.01 {grammar}"):
                got = request(proc, prompt, len(expected), f"ppl=0 ppl_force=1 ppl_score=0 {keys}")
                if got != expected:
                    raise RuntimeError(f"forced replay differs with {keys!r}: {got!r} != {expected!r}")
            if rows.exists() and rows.stat().st_size:
                raise RuntimeError("ppl_score=0 wrote scored rows")
            proc.stdin.write("QUIT\n")
            proc.stdin.flush()
            proc.wait(timeout=120)
            if proc.returncode:
                log.seek(0)
                raise RuntimeError(f"engine exited with {proc.returncode}: {log.read()}")
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=120)
    print("ppl_force_grammar_test: fixed reference preserved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
