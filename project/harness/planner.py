"""Features, a-priori cost model and greedy knapsack ordering for LLM-path tasks.

Tasks solved by template solvers cost 0 and never reach the planner. The priors below
are placeholders calibrated later from real trajectories (priority 2/3); the interface
(features -> est_cost / p_success) is what the trained Ridge/GBM will plug into.
value = P(success) * reward / E[tokens]; select greedily until (1-reserve) of the budget.
"""
import os
import re

from .context import build_context

PRIOR_OUT = {"coding": 450, "data_analysis": 250, "math": 150, "search": 250}
PRIOR_P = {"coding": 0.55, "data_analysis": 0.7, "math": 0.8, "search": 0.7}
SYSTEM_TOKENS = 70  # our own tiny system prompt
RETRY_FACTOR = 1.3  # expected extra cost of the (at most one) retry


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


def est_cost(f: dict) -> float:
    in_tok = SYSTEM_TOKENS + (f["instr_chars"] + f["ctx_chars"]) / 3.0
    out = PRIOR_OUT.get(f["type"], 300)
    return (in_tok / 10.0 + out) * RETRY_FACTOR


def p_success(f: dict, p_feasible: float = 1.0) -> float:
    return PRIOR_P.get(f["type"], 0.4) * p_feasible


def rank(items, budget_left: float, reserve: float = 0.15):
    """items: list of (task, p_feasible). Returns plan rows in execution order."""
    rows = []
    for task, pf in items:
        f = features(task)
        est, p = est_cost(f), p_success(f, pf)
        rows.append({"task": task, "est": est, "p": p, "ratio": p * task["reward"] / est})
    rows.sort(key=lambda r: r["ratio"], reverse=True)
    cap, cum = budget_left * (1 - reserve), 0.0
    for r in rows:
        cum += r["est"]
        r["selected"] = cum <= cap
    return [r for r in rows if r["selected"]] + [r for r in rows if not r["selected"]]
