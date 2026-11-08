"""A context overflow on /v1/messages speaks Anthropic's wording, so clients that decide from the text
(Claude Code compacts on "prompt is too long", shortens max_tokens on "input length and `max_tokens` exceed
context limit") act the way they do for Anthropic's own error; the other APIs keep ours."""
from __future__ import annotations

import json
import re
import sys
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.engine.mock import MockEngine  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CTX = 4096


class AnthropicOverflowWording(unittest.TestCase):
    def post(self, path, body):
        tok = ByteTokenizer()
        eng = MockEngine(tok, "xy", max_context=CTX)
        svc = Service(eng, tok, ChatTemplate.from_file(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                         data=json.dumps({"model": "basalt", **body}).encode(),
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.status, json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read().decode())
        finally:
            httpd.shutdown()

    def test_prompt_alone_refused_names_the_anthropic_maximum(self):
        code, body = self.post("/v1/messages", {"messages": [{"role": "user", "content": "x" * 5000}],
                                                "max_tokens": 20})
        self.assertEqual(code, 400)
        text = body["error"]["message"]
        self.assertRegex(text, r"^prompt is too long: \d+ tokens > 4087 maximum; ", text)
        self.assertIn("request exceeds the context window", text)   # our sentence follows

    def test_max_tokens_too_large_uses_the_anthropic_limit_phrase(self):
        code, body = self.post("/v1/messages", {"messages": [{"role": "user", "content": "hi"}],
                                                "max_tokens": 4090})
        self.assertEqual(code, 400)
        text = body["error"]["message"]
        self.assertRegex(text, r"^input length and `max_tokens` exceed context limit: \d+ \+ 4090 > 4088; ", text)
        self.assertIn("request exceeds the context window", text)

    def test_the_openai_apis_keep_our_wording(self):
        code, body = self.post("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                        "max_tokens": 4090})
        self.assertEqual(code, 400)
        text = body["error"]["message"]
        self.assertIn("request exceeds the context window", text)
        self.assertNotIn("prompt is too long", text)
        self.assertNotIn("exceed context limit", text)


if __name__ == "__main__":
    unittest.main()
