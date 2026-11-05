"""A tool call written in code style (`tools.write`) is the declared tool `write`."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.frontend import OutputParser  # noqa: E402

TOOLS = [{"name": "write", "input_schema": {"type": "object",
                                            "properties": {"path": {"type": "string"}}}},
         {"name": "Bash", "input_schema": {"type": "object",
                                           "properties": {"command": {"type": "string"}}}}]


def call_of(text, tools=TOOLS, stream_tools=False):
    p = OutputParser(thinking=False, tools=tools, stream_tools=stream_tools)
    events = p.feed(text, final=True) + p.finish()
    return next((e for e in events if e.kind == "tool_call"), None)


def call(text, **kw):
    ev = call_of(text, **kw)
    return None if ev is None else ev.call


class NamespacedToolCall(unittest.TestCase):
    def body(self, name, pname, value):
        return (f"<tool_call><function={name}><parameter={pname}>\n{value}\n</parameter"
                f"></function></tool_call>")

    def test_code_style_prefix_is_the_declared_tool(self):
        for prefix in ("tools.", "functions.", "default_api."):
            with self.subTest(prefix=prefix):
                c = call(self.body(prefix + "write", "path", "/tmp/x"))
                self.assertIsNotNone(c)
                self.assertEqual(c.name, "write")
                self.assertEqual(c.arguments, {"path": "/tmp/x"})

    def test_declared_name_wins_over_the_alias(self):
        tools = [{"name": "tools.write"}, {"name": "write"}]
        c = call(self.body("tools.write", "path", "x"), tools=tools)
        self.assertEqual(c.name, "tools.write")

    def test_undeclared_prefixed_name_stays_as_written(self):
        c = call(self.body("tools.unknown", "x", "1"))
        self.assertEqual(c.name, "tools.unknown")

    def test_streamed_start_carries_the_declared_name(self):
        p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
        events = p.feed(self.body("functions.Bash", "command", "ls"), final=True) + p.finish()
        starts = [e for e in events if e.kind == "tool_start"]
        calls = [e for e in events if e.kind == "tool_call"]
        self.assertEqual([e.call.name for e in starts], ["Bash"])
        self.assertEqual(calls[-1].call.name, "Bash")

    def test_plain_names_are_untouched(self):
        c = call(self.body("write", "path", "/tmp/y"))
        self.assertEqual(c.name, "write")


if __name__ == "__main__":
    unittest.main()
