"""Measure whether agreement between two models predicts a correct answer.

For every task with a JSON-plan mode (math / data_analysis / search) we ask two
different models for a plan, execute both plans locally with `harness.dsl`, and
compare the resulting answers. Agreement is a free verifier only if it is
predictive, so the report is:

  * accuracy when the models agree,
  * accuracy when they disagree,
  * how often each model fails to produce a runnable plan,
  * tokens spent per task.

Usage:
  python tools/consensus_probe.py /tmp/shadow --models qwen3.8-27b-noreason,qwen3.6-35b-a3b
  python tools/consensus_probe.py /tmp/shadow --limit 24 --budget 30000

Expected answers are read from task.json for scoring only; the probe never
shows them to a model.
"""
import argparse
import collections
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from harness import dsl  # noqa: E402
from harness.agent import out_mult  # noqa: E402
from harness.budget import Budget, TaskMeter  # noqa: E402
from harness.llm_client import LLMClient  # noqa: E402
from harness.tasks import load_tasks  # noqa: E402
from model_matrix import discover, load_env_file, read_meta, stratified  # noqa: E402

PLAN_TYPES = ("math", "data_analysis", "search")


def _norm(value):
    """Compare '16.2', '16.20' and ' 16.2 ' as equal; keep other text as-is."""
    s = str(value).strip()
    try:
        f = float(s)
    except ValueError:
        return s.lower()
    return f"{f:.10g}"


def _same(a, b):
    return a is not None and b is not None and _norm(a) == _norm(b)


def _plan(client, task, spec, model, meter):
    cap = spec.max_tokens if out_mult(model) == 1 else max(1200, int(spec.max_tokens * out_mult(model)))
    content = task["instruction"]
    if spec.context:
        content += "\n\n" + spec.context
    try:
        reply = client.chat([{"role": "system", "content": spec.system},
                             {"role": "user", "content": content}],
                            model, cap, meter, tag=f"{task['id']}#{model}#plan")
    except Exception as e:  # budget/transport: record, do not abort the probe
        return None, f"{type(e).__name__}: {e}"
    ans, err = dsl.execute(task, reply.text)
    return ans, err


def run(tasks_dir, models, limit, budget, out_dir, env_file, include_infeasible=False):
    load_env_file(env_file)
    if not os.environ.get("LLM_BASE_URL") or not os.environ.get("LLM_API_KEY"):
        raise SystemExit(f"Missing LLM_BASE_URL/LLM_API_KEY; put them into {env_file}")
    pool = [r for r in discover(tasks_dir) if r.get("task_type") in PLAN_TYPES]
    if not include_infeasible:
        pool = [r for r in pool if not r.get("_infeasible")]
    rows = stratified(pool, limit)
    tasks = {t["id"]: t for t in load_tasks(tasks_dir)}
    cases = [tasks[m["_id"]] for m in rows if m["_id"] in tasks]
    budget_obj = Budget(budget)
    client = LLMClient(budget_obj, traj_path=os.path.join(out_dir, "calls.jsonl"))
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for task in cases:
        spec = dsl.spec(task)
        if spec is None:
            continue
        meta = read_meta(task["root"])
        expected = meta.get("expected")
        expected = None if expected is None else str(expected)
        answers, errors = {}, {}
        spent0 = budget_obj.spent
        for model in models:
            meter = TaskMeter(budget_obj, 4000)
            answers[model], errors[model] = _plan(client, task, spec, model, meter)
        status = _status(answers, models)
        rec = {"id": task["id"], "type": task["type"], "expected": expected,
               "answers": answers, "errors": errors, "status": status,
               "cost": round(budget_obj.spent - spent0, 1)}
        records.append(rec)
        print(f"{task['id']:32} {status:10} exp={expected!r:14} " +
              " ".join(f"{m.split('/')[-1]}={answers[m]!r}" for m in models))
    return records, budget_obj


def _status(answers, models):
    vals = [answers[m] for m in models]
    if all(v is None for v in vals):
        return "no_plan"
    if vals[0] is not None and vals[1] is not None:
        return "agree" if _same(*vals) else "disagree"
    return "one_plan"


def summarize(records, models, spent):
    bucket = collections.defaultdict(lambda: collections.Counter())
    feasible = [r for r in records if r["expected"] is not None]
    for r in feasible:
        b = bucket[r["status"]]
        b["n"] += 1
        exp = r["expected"]
        first = r["answers"][models[0]]
        if first is not None:
            b["first_correct"] += 1 if _same(first, exp) else 0
        cand = _consensus(r["answers"], models)
        if cand is not None:
            b["cand_correct"] += 1 if _same(cand, exp) else 0
            b["cand_n"] += 1
    print(f"\n== summary (feasible only, n={len(feasible)}) ==")
    for status in sorted(bucket):
        b = bucket[status]
        acc = (b["cand_correct"] / b["cand_n"]) if b["cand_n"] else float("nan")
        print(f"{status:10} n={b['n']:3} first_correct={b['first_correct']:3} "
              f"answered={b['cand_n']:3} accuracy={acc:.2f}")
    n = len(feasible)
    if n:
        print(f"arbiter needed (disagree) rate: {bucket['disagree']['n'] / n:.2f}; "
              f"no plan at all: {bucket['no_plan']['n'] / n:.2f}")
    print(f"tokens spent: {spent:.0f} ({spent / max(n, 1):.0f} per feasible task)")
    print("agreement accuracy above one-plan accuracy => agreement is a usable verifier")


def _consensus(answers, models):
    vals = [answers[m] for m in models]
    if len(vals) >= 2 and vals[0] is not None and vals[1] is not None and _same(vals[0], vals[1]):
        return vals[0]
    if vals[0] is not None and vals[1] is None:
        return vals[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks_dir")
    ap.add_argument("--models", default="qwen3.8-27b-noreason,qwen3.6-35b-a3b")
    ap.add_argument("--limit", type=int, default=24, help="round-robin over plan types; 0 = all")
    ap.add_argument("--budget", type=float, default=30000)
    ap.add_argument("--out", default="/tmp/consensus_probe")
    ap.add_argument("--env-file", default=os.path.join(ROOT, ".env.local"))
    ap.add_argument("--include-infeasible", action="store_true",
                    help="also probe tasks with no answer (they measure skip logic, not consensus)")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if len(models) != 2:
        raise SystemExit("--models needs exactly two models")
    t0 = time.time()
    records, budget = run(args.tasks_dir, models, args.limit, args.budget, args.out,
                          args.env_file, args.include_infeasible)
    summarize(records, models, budget.spent)
    with open(os.path.join(args.out, "records.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=1, ensure_ascii=False)
    print(f"wall {time.time() - t0:.1f}s, records -> {args.out}/records.json")


if __name__ == "__main__":
    main()
