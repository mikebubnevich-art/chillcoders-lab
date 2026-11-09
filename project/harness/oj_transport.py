"""Optional thin openJiuwen transport for LLM calls.

The harness keeps its own budget, deadline and retry logic; openJiuwen is used
only as the HTTP transport (`OpenAIModelClient`). This keeps the competition
framing ("a harness on top of openJiuwen") real without putting the framework's
runner, agents, memory or context engine in the hot path.

Design constraints:
- no import at module load: openjiuwen is heavy and absent in small dev venvs;
- one process-wide logging config pointing at a writable temp dir, because the
  competition container runs read-only and the default `./logs/` would fail;
- a fresh async HTTP client per call (`use_shared_llm_http_client=False`),
  because calls run in worker threads with separate event loops;
- any failure raises `OjError`; `LLMClient` falls back to raw urllib and
  disables this transport for the rest of the run.

The adapter is pinned to `openjiuwen==0.1.19` (see requirements.txt) and is
covered by tests/test_oj_transport.py plus the full-runtime e2e case.
"""
import asyncio
import os
import tempfile
import threading


class OjUnavailable(RuntimeError):
    """openjiuwen is missing or cannot be configured."""


class OjError(RuntimeError):
    """openjiuwen was configured but the request failed."""


_LOCK = threading.Lock()
_STATE = {"configured": False, "clients": {}}


def available() -> bool:
    try:
        import openjiuwen  # noqa: F401
        return True
    except Exception:
        return False


def _configure() -> None:
    """Configure openjiuwen logging once. Caller holds _LOCK."""
    if _STATE["configured"]:
        return
    from openjiuwen.core.common.logging.log_config import configure_log_config

    log_dir = os.path.join(tempfile.gettempdir(), "jiuwen_logs")
    os.makedirs(log_dir, exist_ok=True)
    configure_log_config({
        "backend": "default",
        "level": "ERROR",
        "structured_output_format": "json",
        "backup_count": 1,
        "max_bytes": 1048576,
        "format": "%(asctime)s | %(log_type)s | %(trace_id)s | %(levelname)s | %(message)s",
        "log_path": log_dir.rstrip(os.sep) + os.sep,
        "log_file": "run/jiuwen.log",
        "propagate": False,
        "output": ["console"],
        "interface_log_file": "interface/jiuwen_interface.log",
        "interface_output": ["console"],
        "prompt_builder_interface_log_file": "interface/jiuwen_prompt_builder_interface.log",
        "performance_log_file": "performance/jiuwen_performance.log",
        "performance_output": ["console"],
        "loggers": {},
    })
    _STATE["configured"] = True


def _client(model: str, base_url: str, api_key: str):
    """Return a cached client. Caller holds _LOCK."""
    key = (base_url, api_key, model)
    client = _STATE["clients"].get(key)
    if client is not None:
        return client
    from openjiuwen.core.foundation.llm.schema.config import (
        ModelClientConfig,
        ModelRequestConfig,
    )
    from openjiuwen.core.foundation.llm.model_clients.openai_model_client import (
        OpenAIModelClient,
    )

    cfg = ModelClientConfig(
        client_provider="OpenAI",
        api_key=api_key,
        api_base=base_url,
        timeout=360.0,
        max_retries=0,
        use_shared_llm_http_client=False,
    )
    client = OpenAIModelClient(
        ModelRequestConfig(model_name=model, temperature=0), cfg
    )
    _STATE["clients"][key] = client
    return client


def warmup(model: str, base_url: str, api_key: str) -> bool:
    """Import and construct the client before the worker pool starts."""
    if not base_url or not api_key:
        return False
    try:
        with _LOCK:
            _configure()
            _client(model, base_url, api_key)
        return True
    except Exception:
        return False


def _num(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def usage_of(message) -> dict:
    """Map openJiuwen UsageMetadata to the OpenAI-style usage dict we price."""
    usage = getattr(message, "usage_metadata", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            usage = usage.model_dump()
        except Exception:
            return {}
    if not isinstance(usage, dict):
        # duck-typed metadata object (tests, older openjiuwen builds)
        usage = {k: getattr(usage, k, 0) for k in
                 ("input_tokens", "output_tokens", "prompt_tokens",
                  "completion_tokens", "reasoning_tokens")}
    prompt = _num(usage.get("input_tokens") or usage.get("prompt_tokens"))
    completion = _num(usage.get("output_tokens") or usage.get("completion_tokens"))
    reasoning = _num(usage.get("reasoning_tokens"))
    if not prompt and not completion and not reasoning:
        return {}
    out = {"prompt_tokens": prompt, "completion_tokens": completion}
    if reasoning:
        out["reasoning_tokens"] = reasoning
    return out


def chat(messages, model: str, max_tokens: int, timeout: float,
         base_url: str, api_key: str):
    """One blocking call, runnable from a worker thread.

    Returns (text, usage). Raises OjUnavailable when the framework cannot be
    used at all, OjError when a configured request fails.
    """
    if not base_url or not api_key:
        raise OjUnavailable("LLM_BASE_URL/LLM_API_KEY are not set")
    try:
        with _LOCK:
            _configure()
            client = _client(model, base_url, api_key)
    except ImportError as e:
        raise OjUnavailable(f"openjiuwen is not importable: {e}") from e
    except Exception as e:
        raise OjUnavailable(f"openjiuwen setup failed: {type(e).__name__}: {e}") from e

    try:
        # Every worker thread has its own event loop; the client is built for
        # this call and closed by openjiuwen's finally block.
        message = asyncio.run(client.invoke(
            messages, max_tokens=max_tokens, temperature=0, timeout=timeout))
    except Exception as e:
        raise OjError(f"{type(e).__name__}: {e}") from e

    text = getattr(message, "content", "") or ""
    if not isinstance(text, str):
        text = str(text)
    return text, usage_of(message)
