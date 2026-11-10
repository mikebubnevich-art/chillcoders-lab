"""OpenAI-compatible chat client with exact token accounting.

Transport is openJiuwen's `OpenAIModelClient` when available (thin adapter in
harness/oj_transport.py); raw urllib is the always-available fallback. The
framework is not allowed to add system prompts/tool schemas, so the "empty call"
cost stays our own tiny prompt.
cost = output + reasoning + input/10 ; provider cache is ignored (as in rules).
"""
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import oj_transport
from .budget import Budget, BudgetExceeded, TaskMeter

try:
    from .local_runtime import apply_local_runtime  # local-only, gitignored file on this machine
    apply_local_runtime()
except ImportError:
    pass

CHARS_PER_TOKEN = 3.0  # conservative (code / Cyrillic) for pre-call reservation


def _off(value: str) -> bool:
    return str(value or "").strip().lower() in ("0", "false", "no", "off")


def _transient(err: str) -> bool:
    e = str(err or "").lower()
    return any(k in e for k in ("429", "rate limit", "ratelimit", "too many",
                                "timeout", "timed out", "500", "502", "503", "504"))


class _Limiter:
    """Adaptive in-flight limit, shared by all worker threads.

    The competition allows 5 sessions, but a proxy tariff may allow fewer
    simultaneous requests. Start at LLM_CONCURRENCY (default 5) and lower the
    limit by one on every 429, so the run adapts instead of burning retries.
    """

    def __init__(self, limit: int):
        self.limit = max(1, int(limit))
        self.floor = max(1, int(os.environ.get("LLM_MIN_CONCURRENCY", "1")))
        self.inflight = 0
        self._cond = threading.Condition()

    def acquire(self, deadline=None) -> None:
        with self._cond:
            while self.inflight >= self.limit:
                left = 1.0 if deadline is None else deadline - time.time()
                if left <= 0:
                    raise RuntimeError("deadline exceeded while waiting for a request slot")
                self._cond.wait(timeout=min(1.0, left))
            self.inflight += 1

    def release(self) -> None:
        with self._cond:
            self.inflight = max(0, self.inflight - 1)
            self._cond.notify()

    def penalize(self) -> None:
        with self._cond:
            if self.limit > self.floor:
                self.limit -= 1
                self._cond.notify_all()


def est_tokens(text: str) -> float:
    return len(text) / CHARS_PER_TOKEN


def cost_of(usage: dict) -> float:
    if not isinstance(usage, dict):
        usage = {}
    prompt = float(usage.get("prompt_tokens") or 0)
    out = float(usage.get("completion_tokens") or 0)
    details = usage.get("completion_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    reasoning = float(details.get("reasoning_tokens") or usage.get("reasoning_tokens") or 0)
    if reasoning > out:  # provider reports reasoning separately -> add it
        out += reasoning
    return out + prompt / 10.0


@dataclass
class Reply:
    text: str
    cost: float
    usage: dict
    transport: str = "urllib"


class LLMClient:
    def __init__(self, budget: Budget, base_url=None, api_key=None, traj_path=None, deadline=None):
        self.budget = budget
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.traj_path = traj_path
        self.deadline = deadline  # absolute time.time(); None = no wall deadline
        self._tlock = threading.Lock()
        self.calls = 0
        self._limiter = _Limiter(int(os.environ.get("LLM_CONCURRENCY", "5")))
        # openJiuwen is the preferred transport. Permanent failures (import,
        # schema, protocol) disable it for the run; transient ones (429,
        # timeout, 5xx) keep it enabled and are retried.
        self.oj_enabled = not _off(os.environ.get("USE_OJIWEN", "1"))
        self.oj_error = None
        self._oj_fail_streak = 0
        self._transport_logged = None
        self._provider_health = {"openjiuwen": "healthy", "urllib": "healthy"}
        self._provider_backoff_until = 0.0

    def _provider_backoff(self, provider: str, transient: bool):
        if not transient:
            return
        self._provider_health[provider] = "degraded"
        self._oj_fail_streak = max(self._oj_fail_streak, 1 if provider == "openjiuwen" else 0)
        backoff = min(30.0, 2.0 ** max(0, self._oj_fail_streak - 1))
        self._provider_backoff_until = max(self._provider_backoff_until, time.time() + backoff)

    def _provider_ok(self) -> bool:
        return time.time() >= self._provider_backoff_until

    def _log(self, rec: dict) -> None:
        if not self.traj_path:
            return
        try:
            with self._tlock, open(self.traj_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def log_task(self, rec: dict) -> None:
        """Append a task-level telemetry row ({type,cost,success,src,...}).

        Per-call rows above describe individual requests; planner calibration
        needs one row per task, otherwise a retried task looks like two tasks.
        """
        self._log({"t": time.time(), **rec})

    def _remaining(self):
        return None if self.deadline is None else self.deadline - time.time()

    def chat(self, messages, model, max_tokens, meter: TaskMeter, tag="") -> Reply:
        t_start = time.time()
        rem = self._remaining()
        if rem is not None and rem <= 5:
            raise RuntimeError("deadline exceeded")
        if not self._provider_ok():
            raise RuntimeError("provider backoff active")
        in_est = sum(est_tokens(m["content"]) + 4 for m in messages)
        worst = in_est / 10.0 + max_tokens
        if worst > meter.room():
            raise BudgetExceeded(f"task/global room {meter.room():.0f} < worst {worst:.0f}")
        self.budget.reserve(worst)
        body = json.dumps({"model": model, "messages": messages,
                           "max_tokens": max_tokens, "temperature": 0}).encode()
        url = self.base_url + "/chat/completions"
        timeout = 180.0
        if rem is not None:
            timeout = max(5.0, min(timeout, rem - 5))

        reply = None
        err = None
        oj = self.oj_enabled
        attempts = max(2, int(os.environ.get("LLM_ATTEMPTS", "4")))
        try:
            for attempt in range(attempts):
                rem = self._remaining()
                if rem is not None and rem <= 5:
                    err = err or "deadline exceeded"
                    break
                timeout = 180.0 if rem is None else max(5.0, min(180.0, rem - 5))
                retry_after = 0.0
                self._limiter.acquire(self.deadline)
                try:
                    if oj:
                        try:
                            text, usage = oj_transport.chat(
                                messages, model, max_tokens, timeout,
                                self.base_url, self.api_key)
                            self._oj_fail_streak = 0
                            reply = (text, usage, "openjiuwen")
                            break
                        except Exception as e:
                            msg = f"{type(e).__name__}: {e}"[:300]
                            self.oj_error = msg
                            err = f"openjiuwen: {msg}"
                            if _transient(msg):
                                # A busy proxy must not disable the adapter, but it
                                # must trigger a cooling window so the next attempt does
                                # not hammer the same provider with repeated retries.
                                self._provider_backoff("openjiuwen", True)
                                self._limiter.penalize()
                                self._oj_fail_streak += 1
                                if self._oj_fail_streak >= 3:
                                    oj = False
                                    self.oj_enabled = False
                                    err += " (disabled after 3 transient failures)"
                            else:
                                self._provider_backoff("openjiuwen", False)
                                oj = False
                                self.oj_enabled = False
                                err += " (adapter disabled)"
                            # fall through to urllib in the same attempt

                    req = urllib.request.Request(url, data=body, headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"})
                    try:
                        with urllib.request.urlopen(req, timeout=timeout) as r:
                            # A malformed body is a retryable provider failure,
                            # not a crash: phase2 must get None, not a traceback.
                            data = json.loads(r.read().decode())
                        if not isinstance(data, dict):
                            err = "response is not a JSON object"
                        else:
                            usage = data.get("usage") or {}
                            choices = data.get("choices") or []
                            first = choices[0] if choices and isinstance(choices[0], dict) else {}
                            msg = first.get("message")
                            text = (msg.get("content") if isinstance(msg, dict) else "") or ""
                            reply = (text, usage, "urllib")
                        break
                    except urllib.error.HTTPError as e:
                        err = f"HTTP {e.code}"
                        if e.code == 429:
                            self._provider_backoff("urllib", True)
                            self._limiter.penalize()
                            ra = e.headers.get("Retry-After") if e.headers else None
                            try:
                                retry_after = max(0.0, float(ra))
                            except (TypeError, ValueError):
                                retry_after = 0.0
                        elif e.code in (500, 502, 503, 504):
                            self._provider_backoff("urllib", True)
                            retry_after = 1.0
                        else:
                            self._provider_backoff("urllib", False)
                            break
                    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                        self._provider_backoff("urllib", True)
                        err = repr(e)
                finally:
                    self._limiter.release()

                pause = max(retry_after, 1.5 * (attempt + 1))
                rem = self._remaining()
                if rem is not None and rem <= pause + 5:
                    break
                time.sleep(pause)
        except Exception:
            self.budget.commit(worst, 0.0)
            raise

        if reply is None:
            # a failed request is assumed free; worst-case is released
            self.budget.commit(worst, 0.0)
            detail = err or "unknown"
            if self.oj_error and not detail.startswith("openjiuwen:"):
                detail = f"{detail} (openjiuwen: {self.oj_error})"
            raise RuntimeError(f"LLM request failed: {detail}")

        text, usage, transport = reply
        with self._tlock:
            if self._transport_logged != transport:
                self._transport_logged = transport
                note = f" ({self.oj_error})" if transport == "urllib" and self.oj_error else ""
                print(f"[transport] {transport}{note}", flush=True)
        if not usage:  # no usage returned: fall back to the conservative estimate
            usage = {"prompt_tokens": in_est, "completion_tokens": est_tokens(text)}
        cost = cost_of(usage)
        self.budget.commit(worst, cost)
        meter.spent += cost
        self.calls += 1
        self._log({"t": time.time(), "tag": tag, "model": model, "usage": usage,
                   "cost": round(cost, 2), "max_tokens": max_tokens,
                   "transport": transport, "elapsed": round(time.time() - t_start, 2),
                   "inflight_limit": self._limiter.limit})
        return Reply(text, cost, usage, transport)
