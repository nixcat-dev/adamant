# Adaptive expert tier

[简体中文](../zh/configuration/adaptive.md)

Flash-Next only. The GPUs hold part of the experts. The adaptive tier counts what the conversation routes to and, in rounds beside decode, moves the most
used experts into VRAM. The defaults are the fastest measured here.

## How a round works

Routing counts rank the experts not in VRAM. A round swaps the most-routed ones in for the least-routed resident ones. A server round
takes three windows: copy out, copy in on a side stream, then use. Counts decay after each round. On the main card every layer ranks
together; on a peer card the main card takes its share first. Moving experts changes rounding, so output with adaptation on is not
bit-identical run to run.

## Settings

| Setting | Default | Trade-off |
| --- | --- | --- |
| `adapt-every` | 1 on a server, 4 with a layer split | Fewer, later rounds cost less link, but a topic change or a fresh engine waits N windows. |
| `adapt-swaps` | 96 | Bigger rounds catch up faster and move more bytes (about 2.6 MiB per NVFP4 expert each way). The RAM copy keeps four times N free places. |
| `peer-adapt-swaps` | `adapt-swaps` | Same, on the peer's link. `0` freezes the peer tier. |
| `adapt-decay` | 0.85 (0.7 with a layer split) | Lower follows topic changes sooner. Higher keeps a long-run placement. |
| `expert-profile-save` | off | Saves the placement; a continuing workload starts on it, a different one starts wrong. |

`hit=` in the `done` line counts only the main card's VRAM lookups, so a higher `hit` is not always faster. Compare `tg`.

[basalt-probe](../probe.md) tries these values on your machine.
