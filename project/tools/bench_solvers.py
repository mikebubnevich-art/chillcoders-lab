"""Phase-1 (0-token) benchmark: coverage, WRONG answers, infeasible detection.

Reads `expected` from task.json (the harness never does). A wrong answer is the
critical metric: it means a solver confidently wrote something that would score
0 and also blocked the LLM fallback.

Usage:
  python tools/gen_synthetic.py /tmp/synth100
  python tools/bench_solvers.py /tmp/synth100 --min-coverage 0.7
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

from harness.tasks import load_tasks  # noqa: E402
from harness.triage import triage  # noqa: E402


def meta_of(task):
    with open(os.path.join(task["dir"], "task.json"), encoding="utf-8") as f:
        return json.load(f)


def run(tasks_dir):
    tasks = load_tasks(tasks_dir)
    rows = []
    t0 = time.time()
    for t in tasks:
        meta = meta_of(t)
        exp = meta.get("expected")
        exp = None if exp is None else str(exp)
        try:
            r = triage(t)
            status, answer = r.status, r.answer
        except Exception as e:  # never let one bad task stop the benchmark
            status, answer = "error", f"{type(e).__name__}: {e}"
        rows.append({"id": t["id"], "type": t["type"], "expected": exp,
                     "status": status, "answer": answer})
    return rows, time.time() - t0


def summarize(rows):
    per = collections.defaultdict(lambda: collections.Counter())
    overall = collections.Counter()
    for r in rows:
        c = per[r["type"]]
        feasible = r["expected"] is not None
        c["total"] += 1
        overall["total"] += 1
        if not feasible:
            c["infeasible_total"] += 1
            overall["infeasible_total"] += 1
            if r["status"] == "infeasible":
                c["infeasible_caught"] += 1
                overall["infeasible_caught"] += 1
            else:
                c["infeasible_missed"] += 1
                overall["infeasible_missed"] += 1
            continue
        c["feasible_total"] += 1
        overall["feasible_total"] += 1
        if r["status"] == "solved":
            c["solved"] += 1
            overall["solved"] += 1
            if r["answer"] == r["expected"]:
                c["correct"] += 1
                overall["correct"] += 1
            else:
                c["wrong"] += 1
                overall["wrong"] += 1
        elif r["status"] == "infeasible":
            c["false_skip"] += 1
            overall["false_skip"] += 1
        else:
            c["missed"] += 1
            overall["missed"] += 1
    return per, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks_dir")
    ap.add_argument("--min-coverage", type=float, default=0.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    rows, sec = run(a.tasks_dir)
    per, ov = summarize(rows)
    if a.json:
        print(json.dumps({"overall": dict(ov), "per_type": {k: dict(v) for k, v in per.items()},
                          "rows": rows, "sec": sec}, indent=1))
    else:
        print(f"{'type':<16}{'total':>6}{'feas':>6}{'solved':>7}{'wrong':>6}{'miss':>6}"
              f"{'inf':>5}{'caught':>7}")
        for typ in sorted(per):
            c = per[typ]
            print(f"{typ:<16}{c['total']:>6}{c['feasible_total']:>6}{c['solved']:>7}"
                  f"{c['wrong']:>6}{c['missed']:>6}{c['infeasible_total']:>5}"
                  f"{c['infeasible_caught']:>7}")
        print(f"{'TOTAL':<16}{ov['total']:>6}{ov['feasible_total']:>6}{ov['solved']:>7}"
              f"{ov['wrong']:>6}{ov['missed']:>6}{ov['infeasible_total']:>5}"
              f"{ov['infeasible_caught']:>7}")
        cov = ov["correct"] / max(1, ov["feasible_total"])
        wr = ov["wrong"] / max(1, ov["solved"])
        ic = ov["infeasible_caught"] / max(1, ov["infeasible_total"])
        print(f"\ncoverage (correct/feasible): {cov:.1%}")
        print(f"wrong-answer rate (wrong/solved): {wr:.1%}")
        print(f"infeasible caught: {ic:.1%}  false skips: {ov['false_skip']}")
        print(f"time: {sec:.1f}s")
        for r in rows:
            if r["expected"] is not None and r["status"] == "solved" and r["answer"] != r["expected"]:
                print(f"WRONG {r['id']}: got {r['answer']!r}, expected {r['expected']!r}")
            elif r["expected"] is not None and r["status"] == "infeasible":
                print(f"FALSE-SKIP {r['id']}")
    bad = ov["wrong"] + ov["false_skip"] > 0
    cov = ov["correct"] / max(1, ov["feasible_total"])
    if cov < a.min_coverage:
        bad = True
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
