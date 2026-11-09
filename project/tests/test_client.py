import json
import os
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness.budget import Budget, TaskMeter
from harness.llm_client import LLMClient


class _BadJSON(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = b"this is not json"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class T(unittest.TestCase):
    def test_past_deadline_fails_before_reserving(self):
        b = Budget(1000)
        c = LLMClient(b, base_url="http://127.0.0.1:1/v1", deadline=time.time() - 1)
        with self.assertRaises(RuntimeError):
            c.chat([{"role": "user", "content": "hi"}], "m", 10, TaskMeter(b, 100))
        self.assertEqual(b.snapshot()["reserved"], 0.0)
        self.assertEqual(b.snapshot()["spent"], 0.0)

    def test_no_deadline_keeps_client_usable(self):
        b = Budget(1000)
        c = LLMClient(b, base_url="http://127.0.0.1:1/v1")
        self.assertIsNone(c.deadline)

    def test_malformed_json_is_a_runtime_error_and_releases_budget(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), _BadJSON)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            b = Budget(1000)
            # a near deadline makes the client stop after the first retry pause
            c = LLMClient(b, base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                          deadline=time.time() + 6)
            with self.assertRaises(RuntimeError):
                c.chat([{"role": "user", "content": "hi"}], "m", 10, TaskMeter(b, 100))
            snap = b.snapshot()
            self.assertEqual(snap["reserved"], 0.0)
            self.assertEqual(snap["spent"], 0.0)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_non_object_json_response_fails_cleanly(self):
        class _ListJSON(_BadJSON):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                body = b"[1, 2, 3]"
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _ListJSON)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            b = Budget(1000)
            c = LLMClient(b, base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                          deadline=time.time() + 6)
            with self.assertRaises(RuntimeError):
                c.chat([{"role": "user", "content": "hi"}], "m", 10, TaskMeter(b, 100))
            self.assertEqual(b.snapshot()["reserved"], 0.0)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
