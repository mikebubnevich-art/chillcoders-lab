"""Minimal OpenAI-compatible chat client (stdlib only) with exact token accounting.

Why stdlib: no framework system prompt / tool schemas are added to requests, so
the "empty call" cost is just our own (tiny) prompt.
cost = output + reasoning + input/10 ; provider cache is ignored (as in rules).
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from .budget import Budget, BudgetExceeded, TaskMeter

CHARS_PER_TOKEN = 3.0  # conservative (code / Cyrillic) for pre-call reservation


def est_tokens(text: str) -> float:
    return len(text) / CHARS_PER_TOKEN


def cost_of(usage: dict) -> float:
    prompt = float(usage.get("prompt_tokens") or 0)
    out = float(usage.get("completion_tokens") or 0)
    details = usage.get("completion_tokens_details") or {}
    reasoning = float(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
    if reasoning > out:  # provider reports reasoning separately -> add it
        out += reasoning
    return out + prompt / 10.0


@dataclass
class Reply:
    text: str
    cost: float
    usage: dict


class LLMClient:
    def __init__(self, budget: Budget, base_url=None, api_key=None, traj_path=None):
        self.budget = budget
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.traj_path = traj_path
        self._tlock = threading.Lock()
        self.calls = 0

    def _log(self, rec: dict) -> None:
        if not self.traj_path:
            return
        try:
            with self._tlock, open(self.traj_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def chat(self, messages, model, max_tokens, meter: TaskMeter, tag="") -> Reply:
        in_est = sum(est_tokens(m["content"]) + 4 for m in messages)
        worst = in_est / 10.0 + max_tokens
        if worst > meter.room():
            raise BudgetExceeded(f"task/global room {meter.room():.0f} < worst {worst:.0f}")
        self.budget.reserve(worst)
        body = json.dumps({"model": model, "messages": messages,
                           "max_tokens": max_tokens, "temperature": 0}).encode()
        url = self.base_url + "/chat/completions"
        data, err = None, None
        for attempt in range(3):
            req = urllib.request.Request(url, data=body, headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"})
            try:
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.loads(r.read().decode())
                break
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
                if e.code not in (429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                err = repr(e)
            time.sleep(1.5 * (attempt + 1))
        if data is None:
            # a failed request is assumed free; worst-case is released
            self.budget.commit(worst, 0.0)
            raise RuntimeError(f"LLM request failed: {err}")
        usage = data.get("usage") or {}
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if not usage:  # no usage returned: fall back to the conservative estimate
            usage = {"prompt_tokens": in_est, "completion_tokens": est_tokens(text)}
        cost = cost_of(usage)
        self.budget.commit(worst, cost)
        meter.spent += cost
        self.calls += 1
        self._log({"t": time.time(), "tag": tag, "model": model, "usage": usage,
                   "cost": round(cost, 2), "max_tokens": max_tokens})
        return Reply(text, cost, usage)
