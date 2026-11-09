#!/usr/bin/env python3
"""Compare the four competition models on the same task set.

The run is sequential on purpose: each model uses up to --workers (<=5)
sessions, so running models in parallel would break the competition limit.
Every model gets a fresh copy of the tasks; answer.txt from one run never
leaks into another.

Usage:
  cp .env.example .env.local          # then put the real proxy URL/key there
  python tools/model_matrix.py --tasks /tmp/synth100 --budget 100000

  # quick pilot: 8 feasible tasks per model + all infeasible ones
  python tools/model_matrix.py --tasks /tmp/synth100 --limit 8 --budget 30000

Why --no-solvers is the default: the synthetic set is fully solved by the
zero-token solvers, so with solvers enabled all four models tie at 0 tokens.
Disable them to measure the model itself. Solver UNSOLVABLE detection still
runs, so infeasible tasks are not hallucinated. Pass --solvers to measure the
full harness instead.

Concurrency: --workers is the number of sessions (competition limit 5), while
--concurrency caps simultaneous HTTP requests (proxy tariff; default 3).
The client also lowers this limit by one on every 429.

Outputs:
  <out>/<model>/solutions.json, trajectory.jsonl, run.log
  <out>/matrix.json, <out>/matrix.md
"""
import argparse
import collections
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_AGENT = os.path.join(ROOT, "run_agent.py")
MODELS = ["qwen3.8-27b", "qwen3.8-27b-noreason",
          "qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason"]
TASK_RE = re.compile(r"^\[task\]\s+(\S+)\s+(\S+)\s+cost=\s*([\d.]+)\s+sec=\s*([\d.]+)")


def load_env_file(path):
    """Simple KEY=VALUE loader; shell environment wins."""
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def read_meta(task_dir):
    try:
        with open(os.path.join(task_dir, "task.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def discover(tasks_dir):
    rows = []
    for name in sorted(os.listdir(tasks_dir)):
        d = os.path.join(tasks_dir, name)
        if not os.path.isdir(d):
            continue
        tj = os.path.join(d, "task.json")
        if not os.path.isfile(tj):
            alt = os.path.join(d, "workspace", "task.json")
            if os.path.isfile(alt):
                tj = alt
            else:
                continue
        meta = read_meta(os.path.dirname(tj))
        meta["_dir"] = d
        meta["_id"] = meta.get("task_id", name)
        meta["_infeasible"] = meta.get("check") == "infeasible"
        rows.append(meta)
    return rows


def stratified(rows, limit):
    """Round-robin across task types; infeasible tasks are always included."""
    feasible = [r for r in rows if not r["_infeasible"]]
    infeasible = [r for r in rows if r["_infeasible"]]
    if limit <= 0 or limit >= len(feasible):
        return rows
    by_type = collections.defaultdict(list)
    for r in feasible:
        by_type[r.get("task_type", "unknown")].append(r)
    queues = [sorted(v, key=lambda r: r["_id"]) for _, v in sorted(by_type.items())]
    picked = []
    while len(picked) < limit:
        progressed = False
        for q in queues:
            if q and len(picked) < limit:
                picked.append(q.pop(0))
                progressed = True
        if not progressed:
            break
    return picked + infeasible


def copy_tasks(rows, dst):
    os.makedirs(dst, exist_ok=True)
    for r in rows:
        shutil.copytree(r["_dir"], os.path.join(dst, os.path.basename(r["_dir"])))


def safe_name(model):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model)


def p95(values):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(0.95 * (len(values) - 1)))]


def is_correct(meta, answer):
    """Dev-only check against task.json expected. The harness never reads it."""
    expected = meta.get("expected")
    check = str(meta.get("check") or "")
    if expected is None:
        return None
    answer = str(answer).strip()
    expected = str(expected).strip()
    if check == "exact_int":
        try:
            return int(answer) == int(expected)
        except (TypeError, ValueError):
            return answer == expected
    if check == "unittest" or expected == "tests_pass":
        return answer.upper() == "DONE"
    return answer == expected


def run_model(model, rows, args, out_root):
    name = safe_name(model)
    run_dir = os.path.join(out_root, name)
    run_tasks = os.path.join(run_dir, "tasks")
    run_out = os.path.join(run_dir, "out")
    traj = os.path.join(run_dir, "trajectory.jsonl")
    shutil.rmtree(run_dir, ignore_errors=True)
    os.makedirs(run_out, exist_ok=True)
    copy_tasks(rows, run_tasks)

    env = {
        **os.environ,
        "OUT": run_out,
        "TRAJ": traj,
        "BUDGET": str(args.budget),
        "MODEL": model,
        "MODEL_CODING": args.coding_model,
        "CODING_STRATEGY": args.coding_strategy,
        "WORKERS": str(min(5, max(1, args.workers))),
        "LLM_CONCURRENCY": str(max(1, args.concurrency)),
        "REASON_MULT": str(max(1.0, args.reason_mult)),
        "DEADLINE_S": str(args.deadline),
        "DISABLE_SOLVERS": "0" if args.solvers else "1",
    }
    if args.transport == "urllib":
        env["USE_OJIWEN"] = "0"
    cmd = [sys.executable, RUN_AGENT, "--tasks-dir", run_tasks, "--out", run_out,
           "--budget", str(args.budget), "--model", model,
           "--workers", env["WORKERS"], "--deadline", str(args.deadline)]
    if args.coding_model:
        cmd += ["--coding-model", args.coding_model,
                "--coding-strategy", args.coding_strategy]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=args.deadline + 240)
        log = proc.stdout + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        log = (e.stdout or "") + (e.stderr or "")
        rc = -9
    wall = time.time() - t0
    with open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8") as f:
        f.write(log)

    done = None
    for line in log.splitlines():
        if line.startswith("[done] "):
            try:
                done = json.loads(line[len("[done] "):])
            except ValueError:
                pass
    solutions = {}
    sol_path = os.path.join(run_out, "solutions.json")
    if os.path.isfile(sol_path):
        try:
            with open(sol_path, encoding="utf-8") as f:
                solutions = json.load(f)
        except ValueError:
            solutions = {}

    records = []
    if os.path.isfile(traj):
        with open(traj, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass
    calls = [r for r in records if "tag" in r]  # per-request rows; task rows have no tag
    elapsed = [float(r.get("elapsed") or 0) for r in calls if r.get("elapsed")]
    transports = collections.Counter(r.get("transport", "unknown") for r in calls)

    meta = {r["_id"]: r for r in rows}
    task_cost, task_sec, task_src = {}, {}, {}
    for line in log.splitlines():
        m = TASK_RE.match(line)
        if m:
            tid, src, cost, sec = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
            task_cost[tid], task_sec[tid], task_src[tid] = cost, sec, src

    answered = correct = wrong = unknown = 0
    false_positive = 0
    reward = 0.0
    task_correct = {}
    caught_literal = 0
    for tid, answer in solutions.items():
        md = meta.get(tid)
        if md is None:
            answered += 1
            unknown += 1
            continue
        if md["_infeasible"]:
            # Both conventions are correct: skip silently, or write UNSOLVABLE.
            if str(answer).strip().upper() == "UNSOLVABLE":
                caught_literal += 1
            else:
                false_positive += 1
            continue
        answered += 1
        ok = is_correct(md, answer)
        task_correct[tid] = ok
        if ok is None:
            unknown += 1
        elif ok:
            correct += 1
            reward += float(md.get("reward", 0) or 0)
        else:
            wrong += 1

    per_type = collections.defaultdict(lambda: {"tasks": 0, "answered": 0, "correct": 0,
                                                "reward": 0.0, "spent": 0.0})
    for tid, md in meta.items():
        if md["_infeasible"]:
            continue
        bucket = per_type[md.get("task_type", "unknown")]
        bucket["tasks"] += 1
        bucket["spent"] += task_cost.get(tid, 0.0)
        if tid in solutions:
            bucket["answered"] += 1
            if task_correct.get(tid):
                bucket["correct"] += 1
                bucket["reward"] += float(md.get("reward", 0) or 0)

    spent = float(done.get("spent", 0.0)) if done else sum(task_cost.values())
    harness_reward = float(done.get("reward", 0.0)) if done else 0.0
    infeasible_total = sum(1 for r in rows if r["_infeasible"])
    return {
        "model": model,
        "rc": rc,
        "wall_sec": round(wall, 1),
        "done": done,
        "answered": answered,
        "correct": correct,
        "wrong": wrong,
        "unknown": unknown,
        "reward": reward,
        "harness_reward": harness_reward,
        "spent": round(spent, 1),
        "cost_per_solve": round(spent / correct, 1) if correct else None,
        "reward_per_1k": round(reward / spent * 1000, 2) if spent else None,
        "infeasible_total": infeasible_total,
        "infeasible_caught": int(done.get("infeasible", 0) or 0) + caught_literal,
        "infeasible_answered": false_positive,
        "calls": len(calls),
        "p50_ms": round(statistics.median(elapsed) * 1000) if elapsed else None,
        "p95_ms": round(p95(elapsed) * 1000) if elapsed else None,
        "transports": dict(transports),
        "per_type": {k: {"tasks": v["tasks"], "answered": v["answered"],
                         "correct": v["correct"],
                         "reward": round(v["reward"], 1),
                         "spent": round(v["spent"], 1)}
                     for k, v in sorted(per_type.items())},
        "tasks": {tid: {"src": task_src.get(tid), "cost": task_cost.get(tid),
                        "sec": task_sec.get(tid), "correct": task_correct.get(tid)}
                  for tid in meta if tid in task_cost or tid in solutions},
        "run_dir": run_dir,
    }


def write_report(out_root, rows, args, tasks_count):
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, "matrix.json"), "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "tasks": tasks_count, "models": rows}, f,
                  ensure_ascii=False, indent=1)
    lines = [
        "# Model matrix", "",
        f"- tasks: `{args.tasks}` ({tasks_count} after limit)",
        f"- budget per model: {args.budget:.0f} output-equivalent tokens",
        f"- workers: {args.workers}, max requests in flight: {args.concurrency}, "
        f"deadline per model: {args.deadline:.0f}s",
        f"- solvers: {'on' if args.solvers else 'off'}, transport: {args.transport}, "
        f"reasoning cap x{args.reason_mult:g}, coding model: {args.coding_model or '-'} "
        f"({args.coding_strategy})",
        "- correctness is checked against `task.json.expected` (dev tool only; "
        "the harness itself never reads it)",
        "", "| model | ok | wrong | unknown | infeasible FP | reward | spent | cost/ok | reward/1k | wall s | calls | transport | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        transports = " ".join(f"{k}:{v}" for k, v in sorted(r["transports"].items())) or "-"
        lines.append(
            f"| `{r['model']}` | {r['correct']} | {r['wrong']} | {r['unknown']} | "
            f"{r['infeasible_answered']}/{r['infeasible_total']} | {r['reward']:.0f} | "
            f"{r['spent']:.0f} | "
            f"{r['cost_per_solve'] if r['cost_per_solve'] is not None else '-'} | "
            f"{r['reward_per_1k'] if r['reward_per_1k'] is not None else '-'} | "
            f"{r['wall_sec']} | {r['calls']} | {transports} | "
            f"{r['p50_ms'] if r['p50_ms'] is not None else '-'} | "
            f"{r['p95_ms'] if r['p95_ms'] is not None else '-'} |")
    lines += ["", "## By task type", "",
              "| model | type | tasks | answered | ok | reward | spent |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        for typ, v in r["per_type"].items():
            lines.append(f"| `{r['model']}` | {typ} | {v['tasks']} | {v['answered']} | "
                         f"{v['correct']} | {v['reward']:.0f} | {v['spent']:.0f} |")
    lines += ["", "Notes: `spent` follows the competition formula "
              "(output + reasoning + input/10). `reward` counts only correct "
              "answers, `harness_reward` in matrix.json is the harness' own "
              "self-report. `infeasible FP` = infeasible tasks that still got "
              "an answer. `p50/p95` are per LLM call, including retries and the "
              "first-call transport setup.", ""]
    with open(os.path.join(out_root, "matrix.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="task set directory")
    ap.add_argument("--models", default="all",
                    help="comma-separated model ids, or 'all' (the four allowed models)")
    ap.add_argument("--coding-model", default=os.environ.get("MODEL_CODING", ""),
                    help="optional stronger model for coding tasks only")
    ap.add_argument("--coding-strategy", default=os.environ.get("CODING_STRATEGY", "fallback"),
                    choices=("fallback", "always"),
                    help="fallback: base first + coding model on failure; always: coding model first")
    ap.add_argument("--budget", type=float, default=100000, help="per-model token budget")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--concurrency", type=int,
                    default=int(os.environ.get("LLM_CONCURRENCY", "3")),
                    help="max simultaneous HTTP requests (proxy tariff; sessions stay <=workers)")
    ap.add_argument("--reason-mult", type=float,
                    default=float(os.environ.get("REASON_MULT", "4")),
                    help="completion cap multiplier for reasoning (non -noreason) models")
    ap.add_argument("--deadline", type=float, default=900, help="seconds per model")
    ap.add_argument("--limit", type=int, default=0,
                    help="max feasible tasks (stratified by type); 0 = all")
    ap.add_argument("--solvers", action="store_true",
                    help="enable zero-token solvers (default: off, to measure the model)")
    ap.add_argument("--transport", choices=("auto", "urllib"), default="auto",
                    help="auto = openJiuwen with urllib fallback; urllib = force stdlib")
    ap.add_argument("--out", default="/tmp/model_matrix")
    ap.add_argument("--env-file", default=os.path.join(ROOT, ".env.local"),
                    help="KEY=VALUE file with LLM_BASE_URL/LLM_API_KEY")
    args = ap.parse_args()

    load_env_file(args.env_file)
    if not os.environ.get("LLM_BASE_URL") or not os.environ.get("LLM_API_KEY"):
        print("LLM_BASE_URL and LLM_API_KEY are not set.\n"
              f"Put them into {args.env_file}:\n"
              "  LLM_BASE_URL=https://<proxy>/v1\n"
              "  LLM_API_KEY=<key>\n"
              "or export them in the shell. The file is gitignored.", file=sys.stderr)
        return 2
    if not os.path.isdir(args.tasks):
        print(f"tasks dir not found: {args.tasks}", file=sys.stderr)
        return 2

    models = MODELS if args.models.strip().lower() == "all" else [
        m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("no models selected", file=sys.stderr)
        return 2

    rows = stratified(discover(args.tasks), args.limit)
    if not rows:
        print(f"no task.json found under {args.tasks}", file=sys.stderr)
        return 2
    print(f"[matrix] {len(rows)} tasks, {len(models)} models, budget={args.budget:.0f} each",
          flush=True)
    results = []
    for model in models:
        print(f"[matrix] {model} ...", flush=True)
        results.append(run_model(model, rows, args, args.out))
        r = results[-1]
        print(f"[matrix] {model}: ok={r['correct']} wrong={r['wrong']} "
              f"unknown={r['unknown']} infeasible_fp={r['infeasible_answered']} "
              f"reward={r['reward']:.0f} spent={r['spent']:.0f} "
              f"wall={r['wall_sec']:.0f}s rc={r['rc']}", flush=True)
    write_report(args.out, results, args, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
