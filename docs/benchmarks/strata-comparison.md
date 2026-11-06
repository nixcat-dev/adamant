# Adamant and Strata on one RTX 4090

[简体中文](../zh/benchmarks/strata-comparison.md)

Development comparison recorded 2026-11-05 on one RTX 4090 (24 GB) at a
450 W power limit, with 64 GB system RAM and an Intel Core i7-13700K. It compares
the deployed Adamant and Strata configurations using the same UD-IQ4_XS source
family. It is a single-machine run, not a general performance claim or quality
comparison.

## Results

Each cell is the median of three runs, with the observed minimum and maximum in
parentheses. Each request generated 512 tokens, returned HTTP 200, ended at the
token limit, and reported zero cached prompt tokens. Prompt lengths vary by a few
tokens between the API wrappers. Adamant prefill rates and decode rates are the
server's logged rounded values; Strata values are its engine status counters.
Wall time is measured by the client and includes request overhead. It is not
engine decode time or time to first token.

| Context | Fixture | Adamant prefill tok/s | Strata prefill tok/s | Adamant decode tok/s | Strata decode tok/s | Adamant wall s | Strata wall s |
|---|---|---:|---:|---:|---:|---:|---:|
| 4K | Prose | 2,927 (2,798–2,930) | 1,740 (1,659–1,746) | 160.1 (153.5–162.8) | 115.9 (110.7–119.0) | 4.6 (4.6–4.7) | 6.8 (6.8–7.0) |
| 4K | Code | 2,768 (2,747–2,777) | 1,740 (1,738–1,740) | 165.4 (164.2–167.9) | 132.8 (132.3–141.1) | 4.6 (4.5–4.6) | 6.2 (6.0–6.2) |
| 4K | Structured | 2,772 (2,689–2,877) | 1,740 (1,740–1,743) | 145.9 (144.0–152.3) | 108.2 (106.7–113.6) | 5.0 (4.8–5.0) | 7.1 (6.9–7.2) |
| 64K | Prose | 6,153 (5,902–6,157) | 3,558 (3,413–3,560) | 144.9 (134.4–146.8) | 114.7 (113.3–117.5) | 14.0 (13.9–14.7) | 22.5 (22.4–23.4) |
| 64K | Code | 6,149 (6,145–6,152) | 3,557 (3,529–3,560) | 161.2 (160.1–165.4) | 136.5 (121.6–136.9) | 13.6 (13.6–13.7) | 21.8 (21.8–22.5) |
| 64K | Structured | 6,155 (6,151–6,156) | 3,556 (3,536–3,558) | 136.0 (135.6–148.6) | 109.1 (109.0–113.5) | 14.2 (13.9–14.2) | 22.8 (22.6–22.9) |

These results show higher measured prefill and decode rates for Adamant in all
six tested fixture/context pairs on this machine. The comparison does not measure
quality, and outputs were not required to match. Three repeats give a limited
view of run-to-run variability; see the ranges and the separate control runs in
the [raw measurements](strata-comparison-data.csv).

## Method and configuration

Both engines were served in turn through the same local llama-swap OpenAI
endpoint. The GPU limit remained 450 W. Each engine received one load/warm-up
request, excluded from the table, then 18 timed requests: prose, code, and
structured-output fixtures at two prompt lengths, repeated three times. The
exact task templates were not retained. Adamant ran first; Strata ran second.
The two engines were not alternated request by request. Two additional same-task
controls ran after the main arms and are included separately in the CSV, not
pooled with the measured repeats. There was no concurrent benchmark request.
The server was restored to Adamant after the comparison.

The prompts concatenate the checked-in `README.md`, `CONTRIBUTING.md`,
`docs/precision/*`, and `docs/running/*`, with `Source section:` markers. The
4K corpus is 14,683 characters (SHA-256
`4005a780cd1d48665b4434e77e442e00ee7f13651488a204fea2894ec9d2a4af`). The 64K
corpus is the first 240,000 characters of the corresponding longer concatenation
(SHA-256
`cbf1d1480883aeadc21c22f72072a7c8237e6dd03f720a7b6372fb9935a6375d`). The case text was `Benchmark case {nonce}. {task}\n\n` followed by the corpus;
nonces were `nonce-20261105-{engine}-{depth}-{fixture}-r{rep}`. Fixture tasks:

- Prose: “Summarize the main technical ideas and caveats in this material in a concise, coherent explanation.”
- Code: “From the material, write a Python function that validates a JSON object with required fields and returns clear errors. Include type hints and a short example.”
- Structured: “Return a JSON object with keys `summary`, `key_points` (array), and `caveats` (array), based on the material. Output valid JSON only.”

The harness recorded payload hashes. Effective inputs were about 4.1K and 64.2K
tokens; exact API counts are in the CSV. Generation settings were temperature 0,
top-p 1, seed 1, no thinking, and a 512-token cap. Both engines reported zero
cached prompt tokens on every measured request.

| | Adamant | Strata |
|---|---|---|
| Source | `4a0d140e0953` (0.2.1) | `ffa80d73133e` (0.2.10; build version 0.1.56) |
| Engine binary SHA-256 | `5ef85c26af2b…a9f2defc` | `0c143e5281ba…241e91eb23` |
| Runtime | bundled MTP, spec 4, PCIe balance off | DFlash2 draft, spec minimum 0.5, adaptive every 1, PCIe balance on |
| Shared settings | UD-IQ4_XS source family, BF16 PLE, context 262144, int8 KV, automatic expert/prefill placement, GPU 0 | same source family and shared settings |

The source inputs were checked by matching their size and first/last 64 MiB
fingerprints. Adamant runs a converted `.basalt` file; Strata reads the original
three GGUF shards. The shared BF16 PLE data is 102,400,495,616 bytes. The engines
do not use identical active tensor representations or precision paths: some
Strata BF16 tensors are expanded from quantized source tensors, while Adamant
uses native representations. The draft methods and PCIe scheduling also differ.
This is a comparison of deployed configurations, not a controlled kernel-only
comparison; no precision parity or output-quality conclusion follows.

The Strata run encountered low available system memory during deep prefill
(about 237 MiB minimum available memory and 503 MiB swap use in sampled
intervals). This may affect timings. Load and warm-up time is excluded from
request timings; one Strata load/warm-up lasted about 70 seconds. The separate
4K prose control measured 104.4 tok/s on Strata and 160.2 tok/s on Adamant; the
64K code control measured 123.7 and 169.6 tok/s, respectively. These controls
ran after the main arms and after an Adamant reload, so they expose timing
variation but do not correct the medians.
