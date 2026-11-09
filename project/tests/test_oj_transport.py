import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import oj_transport
from harness.budget import Budget, TaskMeter
from harness.llm_client import LLMClient


class _OK(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({
            "choices": [{"message": {"content": "local answer"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class _Server:
    def __enter__(self):
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), _OK)
        self.th = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.th.start()
        return f"http://127.0.0.1:{self.srv.server_port}/v1"

    def __exit__(self, *exc):
        self.srv.shutdown()
        self.srv.server_close()


class Usage(unittest.TestCase):
    def test_maps_usage_metadata(self):
        msg = SimpleNamespace(usage_metadata=SimpleNamespace(
            input_tokens=10, output_tokens=5, reasoning_tokens=2))
        self.assertEqual(oj_transport.usage_of(msg),
                         {"prompt_tokens": 10, "completion_tokens": 5,
                          "reasoning_tokens": 2})

    def test_maps_model_dump(self):
        class U:
            def model_dump(self):
                return {"input_tokens": 7, "output_tokens": 3}
        self.assertEqual(oj_transport.usage_of(SimpleNamespace(usage_metadata=U())),
                         {"prompt_tokens": 7, "completion_tokens": 3})

    def test_missing_and_zero_usage_is_empty(self):
        self.assertEqual(oj_transport.usage_of(SimpleNamespace()), {})
        self.assertEqual(oj_transport.usage_of(SimpleNamespace(
            usage_metadata=SimpleNamespace(input_tokens=0, output_tokens=0))), {})


class Adapter(unittest.TestCase):
    def test_missing_base_url_is_unavailable_not_a_crash(self):
        with self.assertRaises(oj_transport.OjUnavailable):
            oj_transport.chat([{"role": "user", "content": "hi"}], "m", 10, 5, "", "")

    def test_setup_error_is_unavailable(self):
        orig = oj_transport._configure

        def boom():
            raise ImportError("no openjiuwen")

        oj_transport._configure = boom
        try:
            with self.assertRaises(oj_transport.OjUnavailable):
                oj_transport.chat([{"role": "user", "content": "hi"}], "m", 10, 5,
                                  "http://127.0.0.1:1/v1", "k")
        finally:
            oj_transport._configure = orig


class Client(unittest.TestCase):
    def setUp(self):
        self.orig_chat = oj_transport.chat

    def tearDown(self):
        oj_transport.chat = self.orig_chat

    def test_prefers_openjiuwen_transport(self):
        calls = []

        def fake(messages, model, max_tokens, timeout, base_url, api_key):
            calls.append((model, max_tokens))
            return "openjiuwen answer", {"prompt_tokens": 10, "completion_tokens": 5}

        oj_transport.chat = fake
        b = Budget(1000)
        c = LLMClient(b, base_url="http://127.0.0.1:1/v1", api_key="k")
        meter = TaskMeter(b, 100)
        r = c.chat([{"role": "user", "content": "hi"}], "m", 10, meter)
        self.assertEqual(r.text, "openjiuwen answer")
        self.assertEqual(r.transport, "openjiuwen")
        self.assertEqual(r.cost, 6.0)  # 5 output + 10/10 input
        self.assertEqual(calls, [("m", 10)])
        self.assertEqual(b.snapshot()["spent"], 6.0)

    def test_falls_back_to_urllib_and_disables_adapter(self):
        calls = []

        def broken(*args, **kwargs):
            calls.append(1)
            raise oj_transport.OjError("boom")

        oj_transport.chat = broken
        with _Server() as url:
            b = Budget(1000)
            c = LLMClient(b, base_url=url, api_key="k")
            meter = TaskMeter(b, 100)
            r = c.chat([{"role": "user", "content": "hi"}], "m", 10, meter)
            self.assertEqual(r.text, "local answer")
            self.assertEqual(r.transport, "urllib")
            self.assertFalse(c.oj_enabled)
            self.assertIn("OjError", c.oj_error)
            self.assertEqual(len(calls), 1)
            # the next call must not touch the broken adapter again
            r2 = c.chat([{"role": "user", "content": "hi"}], "m", 10, meter)
            self.assertEqual(r2.transport, "urllib")
            self.assertEqual(len(calls), 1)
            self.assertEqual(b.snapshot()["spent"], 12.0)

    def test_rate_limit_keeps_adapter_enabled(self):
        calls = []

        def busy(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise oj_transport.OjError("RateLimitError: 429 too many requests")
            return "openjiuwen answer", {"prompt_tokens": 10, "completion_tokens": 5}

        oj_transport.chat = busy
        with _Server() as url:
            b = Budget(1000)
            c = LLMClient(b, base_url=url, api_key="k")
            meter = TaskMeter(b, 100)
            r = c.chat([{"role": "user", "content": "hi"}], "m", 10, meter)
            self.assertEqual(r.transport, "urllib")
            self.assertTrue(c.oj_enabled)  # 429 is transient
            r2 = c.chat([{"role": "user", "content": "hi"}], "m", 10, meter)
            self.assertEqual(r2.transport, "openjiuwen")
            self.assertEqual(len(calls), 2)
            self.assertLessEqual(c._limiter.limit, 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
