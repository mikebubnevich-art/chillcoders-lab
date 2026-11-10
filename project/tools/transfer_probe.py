"""Measure template transfer: one teacher plan per cluster, literal substitution for the rest.

The competition shows all 100 tasks at once, and similar tasks repeat (on the
public generator: 100 tasks -> 53 normalized signatures). The question is not
whether clusters exist but whether a solved *artifact* transfers:

  * math  - the teacher's operation plan with the numbers substituted,
  * data  - the teacher's JSON plan with quoted values / thresholds substituted,
  * coding- identical test files mean the same fixed module solves every member.

Every transfer is executed locally and scored against `expected`, so a failed
substitution shows up as a wrong answer instead of hiding behind a hand-wave.
Tokens are counted for teacher calls only; students cost 0.

Usage:
  python tools/transfer_probe.py /tmp/shadow
  python tools/transfer_probe.py /tmp/synth100 --model qwen3.8-27b-noreason
"""
import argparse
import collections
import json
import os
import re
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
from model_matrix import load_env_file, read_meta  # noqa: E402

PLAN_TYPES = ("math", "data_analysis")
TAIL_RE = re.compile(r"work only inside the current directory.*$", re.I | re.S)
ROWS_RE = re.compile(r"\(\s*\d+\s+rows?\s*\)", re.I)
DEC_RE = re.compile(r"round(?:ed)?\s+to\s+\d+\s+decimal(?:\s+places?)?", re.I)
QUOTED_RE = re.compile(r"'([^']+)'")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
JSON_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)


def _question(instruction: str) -> str:
    """Instruction without the boilerplate tail, row count and rounding clause."""
    text = TAIL_RE.sub("", instruction)
    text = ROWS_RE.sub(" ", text)
    text = DEC_RE.sub(" ", text)
    return text


def signature(instruction: str) -> str:
    text = TAIL_RE.sub("", instruction)
    text = QUOTED_RE.sub("'<v>'", text)
    text = NUM_RE.sub("#", text)
    text = ROWS_RE.sub("(<n> rows)", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _literals(text: str):
    return QUOTED_RE.findall(text), NUM_RE.findall(text)


def _substitute(plan, from_text: str, to_text: str):
    """Replace plan literals that appear in from_text with the positional value from to_text."""
    from_quoted, from_nums = _literals(from_text)
    to_quoted, to_nums = _literals(to_text)
    used_q, used_n = set(), set()

    def map_quoted(v):
        if v not in from_quoted:
            return v
        idx = next((i for i, x in enumerate(from_quoted) if x == v and i not in used_q), None)
        if idx is None or idx >= len(to_quoted):
            return v
        used_q.add(idx)
        return to_quoted[idx]

    def map_num(v):
        key = str(v)
        if key not in from_nums:
            return v
        idx = next((i for i, x in enumerate(from_nums) if x == key and i not in used_n), None)
        if idx is None or idx >= len(to_nums):
            return v
        used_n.add(idx)
        return float(to_nums[idx]) if ("." in to_nums[idx] or isinstance(v, float)) else int(to_nums[idx])

    def walk(node):
        if isinstance(node, dict):
            return {k: walk(x) for k, x in node.items()}
        if isinstance(node, list):
            return [walk(x) for x in node]
        if isinstance(node, str):
            mapped = map_quoted(node)
            return mapped
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            return map_num(node)
        return node

    return walk(plan)


def _ask_plan(client, task, spec, model, meter):
    cap = spec.max_tokens if out_mult(model) == 1 else max(1200, int(spec.max_tokens * out_mult(model)))
    content = task["instruction"] + ("\n\n" + spec.context if spec.context else "")
    try:
        reply = client.chat([{"role": "system", "content": spec.system},
                             {"role": "user", "content": content}],
                            model, cap, meter, tag=f"{task['id']}#teacher")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    m = JSON_RE.search(reply.text)
    if not m:
        return None, "no json block"
    try:
        return json.loads(m.group(1)), None
    except ValueError as e:
        return None, f"bad json: {e}"


def _run_plan(task, plan):
    text = "```json\n" + json.dumps(plan, ensure_ascii=False) + "\n```"
    return dsl.execute(task, text)


def run(tasks_dir, model, budget, out_dir):
    tasks = load_tasks(tasks_dir)
    clusters = collections.defaultdict(list)
    for t in tasks:
        if t["type"] not in PLAN_TYPES:
            continue
        meta = read_meta(t["root"])
        if meta.get("check") == "infeasible":
            continue
        clusters[signature(t["instruction"])].append((t, meta))

    budget_obj = Budget(budget)
    client = LLMClient(budget_obj, traj_path=os.path.join(out_dir, "calls.jsonl"))
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for sig, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        if len(members) < 2:
            continue
        teacher, tmeta = members[0]
        spec = dsl.spec(teacher)
        if spec is None:
            continue
        spent0 = budget_obj.spent
        plan, err = _ask_plan(client, teacher, spec, model, TaskMeter(budget_obj, 4000))
        cost = budget_obj.spent - spent0
        if plan is None:
            rows.append({"type": teacher["type"], "size": len(members), "teacher": teacher["id"],
                         "error": err, "transferred": 0, "cost": cost})
            continue
        tang, terr = _run_plan(teacher, plan)
        teacher_ok = tang is not None and str(tang).strip() == str(tmeta.get("expected")).strip()
        good = 0
        details = []
        for student, smeta in members[1:]:
            moved = _substitute(plan, teacher["instruction"], student["instruction"])
            ans, serr = _run_plan(student, moved)
            ok = ans is not None and str(ans).strip() == str(smeta.get("expected")).strip()
            good += 1 if ok else 0
            details.append({"id": student["id"], "ok": ok, "got": ans, "expected": smeta.get("expected"),
                            "error": serr})
        rows.append({"type": teacher["type"], "size": len(members), "teacher": teacher["id"],
                     "teacher_answer": tang, "teacher_ok": teacher_ok, "plan": plan,
                     "transferred": good, "students": len(members) - 1,
                     "cost": round(cost, 1), "details": details})
        print(f"{teacher['type']:14} cluster={len(members):2} teacher_ok={str(teacher_ok):5} "
              f"transferred={good}/{len(members) - 1}  {teacher['id']}")
    return rows, budget_obj.spent, len([t for t in tasks if t["type"] in PLAN_TYPES])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks_dir")
    ap.add_argument("--model", default="qwen3.8-27b-noreason")
    ap.add_argument("--budget", type=float, default=30000)
    ap.add_argument("--out", default="/tmp/transfer_probe")
    ap.add_argument("--env-file", default=os.path.join(ROOT, ".env.local"))
    args = ap.parse_args()
    load_env_file(args.env_file)
    if not os.environ.get("LLM_BASE_URL") or not os.environ.get("LLM_API_KEY"):
        raise SystemExit(f"Missing LLM_BASE_URL/LLM_API_KEY; put them into {args.env_file}")
    t0 = time.time()
    rows, spent, total = run(args.tasks_dir, args.model, args.budget, args.out)
    covered = sum(r["size"] for r in rows)
    transferred = sum(r.get("transferred", 0) for r in rows)
    students = sum(r.get("students", 0) for r in rows)
    teachers_ok = sum(1 for r in rows if r.get("teacher_ok"))
    print(f"\n== summary ==")
    print(f"plan tasks {total}, clustered {covered}, clusters {len(rows)}, teachers correct {teachers_ok}")
    print(f"students {students}, transferred correct {transferred} "
          f"({(transferred / students if students else 0):.2f})")
    print(f"tokens: {spent:.0f} for {len(rows)} teacher calls "
          f"({spent / max(len(rows), 1):.0f} per cluster)")
    with open(os.path.join(args.out, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    print(f"wall {time.time() - t0:.1f}s, clusters -> {args.out}/clusters.json")


if __name__ == "__main__":
    main()
