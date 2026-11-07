"""A streaming client that stops reading, without closing its connection, must not hold the engine:
the write that makes no progress ends the request as a disconnect and the next request is served."""
import json
import socket
import sys
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve import handler as handler_mod  # noqa: E402
from serve.engine.mock import MockEngine  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


class StalledClient(unittest.TestCase):
    def test_a_client_that_stops_reading_gives_the_turn_back(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, " ".join(str(i) for i in range(80000)), max_context=2_000_000)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        port = httpd.server_address[1]
        stuck = socket.socket()
        try:
            with mock.patch.object(handler_mod, "STREAM_WRITE_TIMEOUT_S", 0.5):
                stuck.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
                stuck.connect(("127.0.0.1", port))
                body = json.dumps({"model": "basalt", "stream": True, "max_tokens": 400000,
                                   "messages": [{"role": "user", "content": "go"}]}).encode()
                stuck.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                              b"Content-Type: application/json\r\nContent-Length: " + str(len(body)).encode()
                              + b"\r\n\r\n" + body)
                # ... and it never reads.  The server fills the tiny receive buffer and its writes stop making progress.
                end = time.monotonic() + 5
                while time.monotonic() < end and not svc.status.get("busy"):
                    time.sleep(0.01)
                self.assertTrue(svc.status.get("busy"), "the request started")
                end = time.monotonic() + 20
                while time.monotonic() < end and svc.status.get("busy"):
                    time.sleep(0.05)
                self.assertFalse(svc.status.get("busy"), "the turn is still held by a client that does not read")
                req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                             data=json.dumps({"model": "basalt", "max_tokens": 16,
                                                              "messages": [{"role": "user", "content": "go"}]}).encode(),
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    self.assertEqual(r.status, 200)
                    self.assertIn("choices", json.loads(r.read().decode()))
        finally:
            stuck.close()
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
