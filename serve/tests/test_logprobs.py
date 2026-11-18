"""logprobs and top_logprobs on Chat Completions: the checks before streaming, the engine's LP lines, and the reply."""
from __future__ import annotations

import io
import json
import queue
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.api.logprobs import REQUEST_KEY, Logprobs, checked, content, engine_key, take_line  # noqa: E402
from serve.engine.mock import MockEngine  # noqa: E402
from serve.engine.process import ProcessEngine  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.service import REASONING_CLOSE, Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
END = ByteTokenizer.SPECIALS.index("<|im_end|>") + 256
NO_THINKING = {"enable_thinking": False}


class BytesTokenizer(ByteTokenizer):
    """ByteTokenizer with token_bytes, as the real tokenizer has: one byte a token, so a character can be split."""

    def token_bytes(self, t):
        return self.SPECIALS[t - 256].encode() if t >= 256 else bytes([t])


class Checks(unittest.TestCase):
    STARTED = SimpleNamespace(info={"context": 4096, "logprobs": 20})

    def test_what_a_request_asks_for(self):
        self.assertEqual(checked({"logprobs": True}, self.STARTED), Logprobs(0))
        self.assertEqual(checked({"logprobs": True, "top_logprobs": 20}, self.STARTED), Logprobs(20))
        for none in ({}, {"logprobs": False}, {"logprobs": None, "top_logprobs": None}, {"top_logprobs": 0}):
            with self.subTest(req=none):
                self.assertIsNone(checked(none, self.STARTED))

    def test_bad_values_are_named(self):
        bad = {"logprobs": [1, "true", {}], "top_logprobs": [-1, 21, 2.5, True, "3"]}
        for field, values in bad.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, f"^{field}:"):
                    checked({"logprobs": True, field: value}, self.STARTED)
        with self.assertRaisesRegex(ValueError, "needs logprobs true"):
            checked({"top_logprobs": 3}, self.STARTED)

    def test_engine_support(self):
        req = {"logprobs": True, "top_logprobs": 2}
        with self.assertRaisesRegex(ValueError, "cannot report"):
            checked(req, SimpleNamespace(info={"context": 4096}))
        with self.assertRaisesRegex(ValueError, "parallel"):
            checked(req, SimpleNamespace(info={"context": 4096, "logprobs": 20}, batch=2))
        self.assertEqual(checked(req, SimpleNamespace(info={"version": None})), Logprobs(2))   # not started: not judged
        self.assertEqual(checked(req, SimpleNamespace()), Logprobs(2))

    def test_engine_key(self):
        self.assertEqual(engine_key(Logprobs(5)), " logprobs=5")
        self.assertEqual(engine_key(None), "")
        self.assertEqual(engine_key({"top": 5, "rows": []}), "")        # a body's own object is never forwarded
        self.assertEqual(ProcessEngine.sampling_keys({REQUEST_KEY: Logprobs(3), "temperature": 0.5}),
                         " temperature=0.5 logprobs=3")


class Entries(unittest.TestCase):
    def test_the_lines_become_openai_entries(self):
        tok = BytesTokenizer()
        lp = Logprobs(2)
        take_line(lp, ord("h"), "LP -0.250000 104:-0.250000 105:-1.500000")
        take_line(lp, 0xC3, "LP -0.100000 195:-0.100000 104:-3.000000")    # "é" in two tokens
        take_line(lp, 0xA9, "LP 0.000000 169:0.000000 97:-20.000000")
        take_line(lp, END, "LP -0.010000")                                  # the end token
        got = content(tok, lp, {END})
        self.assertEqual([e["token"] for e in got], ["h", "", "é"])       # joined, they spell the text
        self.assertEqual([e["bytes"] for e in got], [[104], [0xC3], [0xA9]])
        self.assertEqual([e["logprob"] for e in got], [-0.25, -0.1, 0.0])
        self.assertEqual(got[0]["top_logprobs"], [{"token": "h", "logprob": -0.25, "bytes": [104]},
                                                  {"token": "i", "logprob": -1.5, "bytes": [105]}])
        self.assertEqual(got[1]["top_logprobs"][0], {"token": "�", "logprob": -0.1, "bytes": [0xC3]})


class EngineLines(unittest.TestCase):
    def generate(self, lines, sampling):
        engine = object.__new__(ProcessEngine)
        engine.proc = mock.Mock()
        engine.proc.poll.return_value = None
        engine.proc.stdin = io.StringIO()
        engine.lines = queue.Queue()
        for line in lines:
            engine.lines.put(line)
        engine.can_stop, engine.silence_s = True, 0
        return engine, list(engine.generate([1, 2], 8, sampling, threading.Event()))

    def test_each_token_waits_for_its_lp_line(self):
        lp = Logprobs(1)
        engine, got = self.generate(["T 104", "LP -0.5 104:-0.5", "T 105", "LP -1.5 104:-0.25", "DONE 0 2 1 0 stop"],
                                    {REQUEST_KEY: lp})
        self.assertEqual(got, [104, 105])
        self.assertEqual(lp.rows, [(104, -0.5, [(104, -0.5)]), (105, -1.5, [(104, -0.25)])])
        self.assertIn(" logprobs=1 ", engine.proc.stdin.getvalue())

    def test_a_token_without_its_line_is_still_sent(self):
        lp = Logprobs(0)
        _, got = self.generate(["T 104", "T 105", "LP -1.0", "DONE 0 2 1 0 stop"], {REQUEST_KEY: lp})
        self.assertEqual(got, [104, 105])
        self.assertEqual(lp.rows, [(105, -1.0, [])])

    def test_no_logprobs_reads_tokens_as_before(self):
        engine, got = self.generate(["T 104", "T 105", "DONE 0 2 1 0 stop"], {})
        self.assertEqual(got, [104, 105])
        self.assertNotIn("logprobs", engine.proc.stdin.getvalue())


class ThinkingEngine(MockEngine):
    """Thinks THOUGHT, then answers ANSWER, with an LP row for every token it sends; a prompt that already ends its
    thinking gets the answer at once."""
    THOUGHT = "Let me think about two plus two. " * 2
    ANSWER = "The answer is 4."

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        done = self.tok.decode(ids).endswith("</think>\n\n")
        text = self.ANSWER if done else self.THOUGHT + "</think>\n\n" + self.ANSWER
        for t in (self.tok.encode(text) + [END])[:max_new]:
            sampling[REQUEST_KEY].rows.append((t, -0.5, []))
            yield t


class Http(unittest.TestCase):
    def start(self, engine):
        self.engine = engine
        self.httpd = serve(Service(engine, engine.tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja")),
                           port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def setUp(self):
        self.start(MockEngine(BytesTokenizer(), "ok", max_context=4096))

    def post(self, path, **extra):
        body = {"model": "basalt", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], **extra}
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode()

    def test_the_reply_lists_every_token(self):
        code, body = self.post("/v1/chat/completions", logprobs=True, top_logprobs=3, chat_template_kwargs=NO_THINKING)
        self.assertEqual(code, 200, body)
        choice = json.loads(body)["choices"][0]
        self.assertEqual(choice["message"]["content"], "ok")
        entries = choice["logprobs"]["content"]
        self.assertEqual([e["token"] for e in entries], ["o", "k"])       # the end token is not listed
        self.assertEqual([e["bytes"] for e in entries], [[111], [107]])
        self.assertEqual([len(e["top_logprobs"]) for e in entries], [3, 3])
        self.assertIsNone(choice["logprobs"]["refusal"])

    def test_the_thinking_is_listed_too(self):
        """With thinking on, the mock's first "ok" ends inside its thinking: the server closes the thinking itself and
        the mock answers "ok".  Both passes are listed; the end token and the server's close are not."""
        code, body = self.post("/v1/chat/completions", logprobs=True)
        self.assertEqual(code, 200, body)
        choice = json.loads(body)["choices"][0]
        self.assertEqual(choice["message"]["content"], "ok")
        self.assertEqual([e["token"] for e in choice["logprobs"]["content"]], ["o", "k", "o", "k"])

    def test_a_stream_carries_them_on_its_last_chunk(self):
        code, body = self.post("/v1/chat/completions", logprobs=True, stream=True, chat_template_kwargs=NO_THINKING)
        self.assertEqual(code, 200, body)
        chunks = [json.loads(x[6:]) for x in body.split("\n\n") if x.startswith("data: {")]
        self.assertEqual([("logprobs" in c["choices"][0]) for c in chunks], [False] * (len(chunks) - 1) + [True])
        self.assertEqual([e["token"] for e in chunks[-1]["choices"][0]["logprobs"]["content"]], ["o", "k"])
        self.assertEqual(chunks[-1]["choices"][0]["logprobs"]["content"][0]["top_logprobs"], [])

    def test_no_logprobs_without_the_field(self):
        for extra in ({}, {"logprobs": False}):
            with self.subTest(extra=extra):
                code, body = self.post("/v1/chat/completions", **extra)
                self.assertEqual(code, 200, body)
                self.assertNotIn("logprobs", json.loads(body)["choices"][0])
                self.assertIsNone(self.engine.requests[-1][1].get(REQUEST_KEY))

    def test_a_bad_field_is_a_400_before_the_stream(self):
        for bad in ({"logprobs": "yes"}, {"logprobs": True, "top_logprobs": 21}, {"top_logprobs": 2}):
            for stream in (False, True):
                with self.subTest(bad=bad, stream=stream):
                    code, body = self.post("/v1/chat/completions", stream=stream, **bad)
                    self.assertEqual(code, 400, body)
                    self.assertIn("logprobs", json.loads(body)["error"]["message"])
        self.assertEqual(self.engine.requests, [])

    def test_other_apis_do_not_forward_it(self):
        body = {"logprobs": True, "_logprobs": {"top": 5, "rows": []}}
        self.assertEqual(self.post("/v1/messages", **body)[0], 200)
        self.assertEqual(self.post("/v1/responses", input="hi", **body)[0], 200)
        self.assertTrue(self.engine.requests)
        for _, sampling in self.engine.requests:
            self.assertNotIn("logprobs", ProcessEngine.sampling_keys(sampling))

    def test_the_tokens_the_server_writes_are_not_listed(self):
        """At the thinking budget the server writes the close of the thinking itself: those tokens have no entry."""
        self.start(ThinkingEngine(BytesTokenizer(), "x", max_context=4096))
        code, body = self.post("/v1/chat/completions", logprobs=True, reasoning_budget_tokens=20, max_tokens=400)
        self.assertEqual(code, 200, body)
        reply = json.loads(body)
        entries = reply["choices"][0]["logprobs"]["content"]
        self.assertEqual("".join(e["token"] for e in entries), ThinkingEngine.THOUGHT[:20] + ThinkingEngine.ANSWER)
        extra = len(BytesTokenizer().encode(REASONING_CLOSE, parse_special=True))
        self.assertEqual(len(entries) + extra + 1, reply["usage"]["completion_tokens"])   # + the end token


if __name__ == "__main__":
    unittest.main()
