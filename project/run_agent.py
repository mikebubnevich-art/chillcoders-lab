#!/usr/bin/env python3
"""Entry point: python /app/run_agent.py --tasks-dir /data/tasks

Phase 1 (0 tokens, parallel): template solvers + feasibility triage for ALL tasks.
Phase 2 (LLM): remaining tasks ordered by the planner (P(success)*reward/E[tokens]).
Env: OUT, TASKS, LLM_BASE_URL, LLM_API_KEY. Optional: BUDGET (default 100000), MODEL,
WORKERS (<=5), TASK_CAP, DEADLINE_S, TRAJ, DISABLE_SOLVERS=1.
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import planner  # noqa: E402
from harness.agent import solve_llm  # noqa: E402
from harness.budget import Budget, TaskMeter  # noqa: E402
from harness.llm_client import LLMClient  # noqa: E402
from harness.storage import Results  # noqa: E402
from harness.tasks import load_tasks  # noqa: E402
from harness.triage import triage  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-dir", default=os.environ.get("TASKS", "/data/tasks"))
    ap.add_argument("--out", default=os.environ.get("OUT"))
    ap.add_argument("--budget", type=float, default=float(os.environ.get("BUDGET", 100000)))
    ap.add_argument("--model", default=os.environ.get("MODEL", "qwen3.6-35b-a3b-noreason"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 5)))
    ap.add_argument("--task-cap", type=float, default=float(os.environ.get("TASK_CAP", 1500)))
    ap.add_argument("--deadline", type=float, default=float(os.environ.get("DEADLINE_S", 3300)))
    a = ap.parse_args()

    t0 = time.time()
    budget = Budget(a.budget)
    client = LLMClient(budget, traj_path=os.environ.get("TRAJ", "/tmp/trajectories.jsonl"))
    results = Results(a.out)
    use_solvers = os.environ.get("DISABLE_SOLVERS") != "1"
    tasks = load_tasks(a.tasks_dir)
    workers = max(1, min(5, a.workers))
    stats = {"solved": 0, "solver": 0, "llm_solved": 0, "infeasible": 0, "failed": 0, "skipped": 0, "reward": 0.0}
    lock = threading.Lock()
    print(f"[run] {len(tasks)} tasks, budget={a.budget:.0f}, model={a.model}", flush=True)

    def bump(**kw):
        with lock:
            for k, v in kw.items():
                stats[k] += v

    # ---- phase 1: free triage ----
    def phase1(t):
        try:
            r = triage(t, use_solvers)
        except Exception as e:
            print(f"[task] {t['id']:<22} triage-error {type(e).__name__}", flush=True)
            return t, None
        if r.status == "solved":
            results.save(t, r.answer)
            bump(solved=1, solver=1, reward=t["reward"])
            print(f"[task] {t['id']:<22} solver    cost=    0.0", flush=True)
        elif r.status == "infeasible":
            bump(infeasible=1)
            print(f"[task] {t['id']:<22} infeasible {r.reasons}", flush=True)
        return t, r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        triaged = list(ex.map(phase1, tasks))
    todo = [(t, r.p_feasible) for t, r in triaged if r is not None and r.status == "llm"]
    todo += [(t, 0.5) for t, r in triaged if r is None]  # triage crashed: let the LLM try

    # ---- phase 2: planned LLM spending ----
    plan = planner.rank(todo, budget.remaining())
    for p in plan:
        print(f"[plan] {p['task']['id']:<22} est={p['est']:6.0f} p={p['p']:.2f} ratio={p['ratio']:.4f} "
              f"{'*' if p['selected'] else ' '}", flush=True)

    def phase2(p):
        t = p["task"]
        if time.time() - t0 > a.deadline:
            bump(skipped=1)
            return
        meter = TaskMeter(budget, a.task_cap)
        try:
            ans, src = solve_llm(t, client, meter, a.model)
        except Exception as e:
            ans, src = None, f"error:{type(e).__name__}"
        if ans:
            results.save(t, ans)
            bump(solved=1, llm_solved=1, reward=t["reward"])
        else:
            bump(failed=1)
        print(f"[task] {t['id']:<22} {src:<9} cost={meter.spent:7.1f} {budget.snapshot()}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(phase2, plan))
    print("[done]", json.dumps({**stats, **budget.snapshot(), "sec": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
