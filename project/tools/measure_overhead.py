"""Measure the cost of an 'empty' call per model (run once with real proxy creds)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.budget import Budget, TaskMeter
from harness.llm_client import LLMClient

MODELS = ["qwen3.8-27b-noreason", "qwen3.6-35b-a3b-noreason", "qwen3.8-27b", "qwen3.6-35b-a3b"]
b = Budget(100000)
c = LLMClient(b)
for m in MODELS:
    try:
        r = c.chat([{"role": "user", "content": "Reply with: 4"}], m, 64, TaskMeter(b, 1e9), tag="overhead")
        print(f"{m:28s} cost={r.cost:6.1f} usage={r.usage}")
    except Exception as e:
        print(m, "ERR", e)
print(b.snapshot())
