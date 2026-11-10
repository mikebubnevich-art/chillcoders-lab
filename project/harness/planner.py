"""Features, a-priori cost model and greedy knapsack ordering for LLM-path tasks.

Tasks solved by template solvers cost 0 and never reach the planner. The priors below
are placeholders calibrated later from real trajectories; the interface
(features -> est_cost / p_success) is what the trained Ridge/GBM will plug into.
value = P(success) * reward / E[tokens]; select greedily until (1-reserve) of the budget.
"""
import json
import os
import re

from .context import build_context


def _coerce_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        return v in {"1", "true", "yes", "y", "ok", "success", "solved"}
    return False


def _task_type(rec):
    typ = rec.get("type") or rec.get("task_type") or rec.get("kind") or rec.get("category") or ""
    if isinstance(typ, str):
        typ = typ.strip()
    if typ:
        return str(typ)
    tag = rec.get("tag") or rec.get("task") or rec.get("name") or ""
    if isinstance(tag, str):
        tag = tag.strip()
        if tag and "#" in tag:
            tag = tag.split("#", 1)[0]
        if tag:
            return tag
    return ""


def _usage_cost(rec):
    usage = rec.get("usage") or rec.get("usage_details") or rec.get("usage_metadata") or {}
    if hasattr(usage, "model_dump"):
        try:
            usage = usage.model_dump()
        except Exception:
            usage = {}
    if not isinstance(usage, dict):
        return None
    normalized = {}
    for key, value in usage.items():
        if key in {"prompt_tokens", "input_tokens"}:
            normalized["prompt_tokens"] = value
        elif key in {"completion_tokens", "output_tokens"}:
            normalized["completion_tokens"] = value
        elif key == "reasoning_tokens":
            normalized["reasoning_tokens"] = value
        else:
            normalized[key] = value
    from .llm_client import cost_of
    try:
        return cost_of(normalized)
    except Exception:
        return None


def _normalize_status(rec):
    for key in ("status", "state", "outcome", "result"):
        value = rec.get(key)
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"solved", "success", "done", "passed", "ok", "succeeded", "true", "1"}:
                return True
            if v in {"failed", "error", "timeout", "retry", "skip", "skipped", "false", "0"}:
                return False
    return None


def _normalize_record(rec):
    if not isinstance(rec, dict):
        return None
    typ = _task_type(rec)
    if not typ:
        return None
    candidates = [
        rec.get("cost"), rec.get("spent"), rec.get("total_cost"), rec.get("actual_cost"),
        rec.get("total"), rec.get("tokens_cost"), rec.get("amount"),
    ]
    cost = None
    for candidate in candidates:
        cost = _coerce_float(candidate)
        if cost is not None:
            break
    if cost is None:
        cost = _usage_cost(rec)
    if cost is None or cost <= 0:
        return None
    ok = _normalize_status(rec)
    if ok is None:
        for key in ("success", "ok", "passed", "succeeded"):
            ok = _as_bool(rec.get(key))
            if rec.get(key) is not None:
                break
    if ok is None:
        ok = False
    reward = _coerce_float(rec.get("reward"))
    return {"type": typ, "cost": float(cost), "success": bool(ok), "reward": reward}

PRIOR_OUT = {"coding": 450, "data_analysis": 250, "math": 150, "search": 250}
PRIOR_P = {"coding": 0.55, "data_analysis": 0.7, "math": 0.8, "search": 0.7}
SYSTEM_TOKENS = 70  # our own tiny system prompt
RETRY_FACTOR = 1.3  # expected extra cost of the (at most one) retry


def load_trajectory_stats_from_rows(rows) -> dict:
    """Aggregate task telemetry from an iterable of JSON/dict records."""
    stats = {}
    for rec in rows:
        item = _normalize_record(rec)
        if item is None:
            continue
        typ = item["type"]
        bucket = stats.setdefault(typ, {"costs": [], "successes": 0, "count": 0})
        bucket["costs"].append(item["cost"])
        bucket["count"] += 1
        if item["success"]:
            bucket["successes"] += 1

    out = {}
    for typ, bucket in stats.items():
        costs = sorted(bucket["costs"])
        count = bucket["count"]
        if not costs or count == 0:
            continue
        p90 = costs[min(len(costs) - 1, int(round(0.9 * (len(costs) - 1))))]
        out[typ] = {
            "avg_cost": sum(costs) / len(costs),
            "p90_cost": p90,
            "success_rate": bucket["successes"] / count,
            "count": count,
        }
    return out


def load_trajectory_stats(path: str | os.PathLike[str]) -> dict:
    """Load historical JSONL telemetry and aggregate by task type.

    Accepted shapes include both task-level events: {"type": "math", "cost": 120.0,
    "success": true} and mixed runtime rows emitted by the client or runner.
    Records with missing or malformed values are ignored.
    """
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return {}
    return load_trajectory_stats_from_rows(rows)


def load_planner_stats(path: str | os.PathLike[str] | None = None) -> dict:
    """Resolve empirical planner stats from an explicit path or environment variables.

    Supported env keys: PLANNER_STATS, TRAJ_STATS, TRAJ. The function prefers the
    first existing file and falls back to an empty dict when nothing is set.
    """
    if path is not None and str(path).strip():
        return load_trajectory_stats(path)

    for key in ("PLANNER_STATS", "TRAJ_STATS", "TRAJ"):
        candidate = os.environ.get(key)
        if not candidate:
            continue
        if os.path.exists(candidate):
            return load_trajectory_stats(candidate)
    return {}


def features(task: dict) -> dict:
    d = task["dir"]
    exts, n_files, size = {}, 0, 0
    for root, _, files in os.walk(d):
        for fn in files:
            if fn == "task.json":
                continue
            n_files += 1
            e = os.path.splitext(fn)[1]
            exts[e] = exts.get(e, 0) + 1
            size += os.path.getsize(os.path.join(root, fn))
    ctx = build_context(d)
    text = task["instruction"]
    return {
        "type": task.get("type", "unknown"), "instr_chars": len(text), "ctx_chars": len(ctx),
        "n_files": n_files, "bytes": size, "n_py": exts.get(".py", 0), "n_csv": exts.get(".csv", 0),
        "n_md": exts.get(".md", 0), "has_tests": int(any(f.startswith("test_") for f in os.listdir(d))),
        "n_numbers": len(re.findall(r"\d+", text)), "reward": task["reward"],
    }


def _empirical_weight(st: dict) -> float:
    """How much to trust an empirical bucket: 0 for one sample, max 0.5 at 10+."""
    try:
        n = max(1, int(st.get("count") or 1))
    except (TypeError, ValueError):
        n = 1
    return min(0.5, n / 20.0)


def est_cost(f: dict, out_mult: float = 1.0, stats: dict | None = None) -> float:
    base = SYSTEM_TOKENS + (f["instr_chars"] + f["ctx_chars"]) / 3.0
    prior = PRIOR_OUT.get(f["type"], 300) * max(1.0, float(out_mult))
    raw = (base / 10.0 + prior) * RETRY_FACTOR

    st = (stats or {}).get(f["type"], {})
    empirical = st.get("p90_cost", st.get("avg_cost"))
    if empirical is None:
        return raw
    # One wild sample must not triple the budget: winsorize the empirical
    # bucket at 3x the a-priori estimate before blending it in.
    empirical = min(float(empirical), 3.0 * raw)
    w = _empirical_weight(st)
    # Conservative: empirical data may raise the estimate but never lower it
    # below the a-priori envelope. Under-reserving is what blows the budget.
    return max(raw, (1.0 - w) * raw + w * float(empirical))


def p_success(f: dict, p_feasible: float = 1.0, stats: dict | None = None) -> float:
    prior = PRIOR_P.get(f["type"], 0.4) * p_feasible
    st = (stats or {}).get(f["type"], {})
    empirical = st.get("success_rate")
    if empirical is None:
        return prior
    w = _empirical_weight(st)
    return max(0.0, min(1.0, (1.0 - w) * prior + w * empirical * p_feasible))


def rank(items, budget_left: float, reserve: float = 0.15, out_mult: float = 1.0,
         out_mult_by_type=None, stats: dict | None = None):
    """items: list of (task, p_feasible). Returns plan rows in execution order."""
    by_type = out_mult_by_type or {}
    rows = []
    for task, pf in items:
        f = features(task)
        est = est_cost(f, by_type.get(f["type"], out_mult), stats=stats)
        p = p_success(f, pf, stats=stats)
        rows.append({"task": task, "est": est, "p": p, "ratio": p * task["reward"] / est})
    rows.sort(key=lambda r: r["ratio"], reverse=True)
    cap, cum = budget_left * (1 - reserve), 0.0
    for r in rows:
        cum += r["est"]
        r["selected"] = cum <= cap
    return [r for r in rows if r["selected"]] + [r for r in rows if not r["selected"]]
