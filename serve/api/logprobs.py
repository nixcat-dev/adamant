"""OpenAI `logprobs` and `top_logprobs` on Chat Completions: checked before any streaming, the engine's `LP` lines kept
with the request, and the reply's `logprobs.content`."""
from __future__ import annotations

from dataclasses import dataclass, field

from serve.util.text import Detokenizer

# the engine-side key of a request; the handler stores the checked request there, so only Chat Completions sets it
REQUEST_KEY = "_logprobs"
MAX_TOP = 20                                   # OpenAI's limit, and the engine's (INFO logprobs=N)


@dataclass
class Logprobs:
    """One request's log-probabilities: `top` alternatives a token, and a row for each token the engine sent, in order:
    (token id, logprob, [(id, logprob), ...] best first)."""
    top: int
    rows: list = field(default_factory=list)


def checked(req: dict, engine) -> Logprobs | None:
    """The request's log-probabilities, or None when it asks for none; ValueError naming the problem.  The engine
    reports its limit as INFO logprobs=N (absent from a started engine: no support).  An engine that has not started
    (lazy load) or has no `info` (the mock) is not judged."""
    want, top = req.get("logprobs"), req.get("top_logprobs")
    if want is not None and not isinstance(want, bool):
        raise ValueError("logprobs: true or false")
    if top is not None and (isinstance(top, bool) or not isinstance(top, int) or not 0 <= top <= MAX_TOP):
        raise ValueError(f"top_logprobs: a whole number from 0 to {MAX_TOP}")
    if top and not want:
        raise ValueError("top_logprobs: needs logprobs true")
    if not want:
        return None
    info = getattr(engine, "info", None)
    if isinstance(info, dict) and "context" in info and not isinstance(info.get("logprobs"), int):
        raise ValueError("logprobs: this engine build cannot report log-probabilities")
    if getattr(engine, "batch", 0):
        raise ValueError("logprobs: not supported with parallel slots (--parallel)")
    return Logprobs(top or 0)


def engine_key(lp) -> str:
    """The engine protocol's ` logprobs=K`, or nothing.  Only the handler's checked request is sent: a request body
    cannot carry a Logprobs."""
    return f" logprobs={lp.top}" if isinstance(lp, Logprobs) else ""


def take_line(lp: Logprobs, token: int, line: str) -> None:
    """Keep the engine's `LP <logprob> [<id>:<logprob> ...]` line of `token`."""
    f = line.split()
    lp.rows.append((token, float(f[1]), [(int(i), float(v)) for i, _, v in (x.partition(":") for x in f[2:])]))


def token_bytes(tok, t: int) -> bytes:
    return tok.token_bytes(t) if hasattr(tok, "token_bytes") else tok.decode([t]).encode("utf-8")


def content(tok, lp: Logprobs, stop_ids) -> list[dict]:
    """The reply's `logprobs.content`: one entry per token the engine sent, in order, the end token left out.  An
    entry's `token` is the text that token completes, so the entries, joined, spell what the model wrote."""
    detok, out = Detokenizer(tok), []
    for t, value, alts in lp.rows:
        if t in stop_ids:
            continue
        out.append({"token": detok.push(t), "logprob": value, "bytes": list(token_bytes(tok, t)),
                    "top_logprobs": [{"token": token_bytes(tok, i).decode("utf-8", "replace"), "logprob": v,
                                      "bytes": list(token_bytes(tok, i))} for i, v in alts]})
    return out
