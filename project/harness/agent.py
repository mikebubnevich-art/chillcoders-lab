"""Minimal one-shot agent.

Flow per task (cheapest first):
  (template solvers + feasibility run earlier in triage, 0 tokens)
  1. ONE LLM call: model writes a python3 script, we run it locally (0 tokens)
  2. at most ONE retry with the (truncated) error output, then give up
Why: history re-sent every step grows quadratically; one script call + one retry
is bounded by 2 calls and the script does the real work for free.

The script runs in a disposable copy of the task dir (harness.runner.Sandbox).
Coding tasks are accepted only when answer.txt == DONE and the original tests
pass; tests are restored from a snapshot before verification.
"""
import os
import re
import shlex
import sys

from . import dsl, patch, runner
from .budget import BudgetExceeded, TaskMeter
from .context import build_context
from .llm_client import LLMClient
from .storage import answer_marker

SYSTEM = ("Solve the task by writing ONE python3 script (stdlib, pandas, sympy allowed). "
          "Reply with a single ```python block and no prose. The script runs in the "
          "task directory and must write ONLY the final answer to answer.txt.")

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
RUN_CMD_RE = re.compile(r"`\s*((?:python3?|pytest)\b[^`\n]*)`", re.I)
RUN_PLAIN_RE = re.compile(r"\bpython3?\s+([\w./-]+\.py)\b", re.I)

MAX_OUT = {"math": 300, "search": 400, "data_analysis": 500, "coding": 900}
CODING_TYPES = ("coding", "code")


def out_mult(model: str) -> float:
    """Reasoning models spend the cap on hidden reasoning first.

    They need a larger completion budget or the call returns empty content at
    `finish_reason=length`. `-noreason` models keep the tight caps.
    """
    if "noreason" in str(model or "").lower():
        return 1.0
    try:
        return max(1.0, float(os.environ.get("REASON_MULT", "4")))
    except (TypeError, ValueError):
        return 4.0


def extract_code(text: str) -> str:
    m = CODE_RE.search(text)
    if m:
        return m.group(1)
    return text if "answer.txt" in text else ""


def model_for(task, base: str, coding: str = "", strategy: str = "always") -> str:
    """Per-task model choice; the rules allow switching on any call.

    strategy=always  -> coding tasks go straight to the coding model;
    strategy=fallback -> start with the base model, retry failures with coding.
    """
    if coding and task.get("type") in CODING_TYPES and strategy != "fallback":
        return coding
    return base


def fallback_model(task, src: str, base: str, coding: str = "",
                   strategy: str = "fallback") -> str:
    """Model for a second, surgical retry of a failed coding task, or ''."""
    if strategy != "fallback" or not coding or coding == base:
        return ""
    if task.get("type") not in CODING_TYPES or src != "failed":
        return ""
    return coding


def task_cap(task, base: float = 1500.0, mult: float = 1.0) -> float:
    """Per-task cap that must fit the first attempt *and* one full retry."""
    out = float(MAX_OUT.get(task.get("type"), 600)) * max(1.0, float(mult))
    return max(float(base), 2.0 * out + 600.0)


def _argv(cmd: str):
    try:
        parts = shlex.split(cmd)
    except ValueError:
        return None
    if not parts:
        return None
    first = os.path.basename(parts[0])
    if first.startswith("python"):
        parts[0] = sys.executable
    elif first == "pytest":
        parts = [sys.executable, "-m", "pytest"] + parts[1:]
    else:
        return None
    return parts


def test_commands(instruction: str, orig_dir: str):
    """Commands from the instruction (`python3 test_x.py`), else test_*.py files."""
    cmds = []
    for m in RUN_CMD_RE.finditer(instruction):
        argv = _argv(m.group(1).strip())
        if argv:
            cmds.append(argv)
    if not cmds:
        for m in RUN_PLAIN_RE.finditer(instruction):
            argv = [sys.executable, m.group(1)]
            if os.path.isfile(os.path.join(orig_dir, m.group(1))):
                cmds.append(argv)
    if not cmds:
        for root, dirs, files in os.walk(orig_dir):
            dirs[:] = [d for d in dirs if not d.startswith((".", "__"))]
            for fn in sorted(files):
                if fn.startswith("test_") and fn.endswith(".py"):
                    cmds.append([sys.executable, os.path.relpath(os.path.join(root, fn), orig_dir)])
            if len(cmds) >= 3:
                break
    seen, out = set(), []
    for c in cmds:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out[:3]


def _verify_coding(sb: runner.Sandbox, instruction: str):
    """(ok, detail). Restores tests first, then runs them in the sandbox."""
    cmds = test_commands(instruction, sb.orig)
    extras = [a for c in cmds for a in c[1:] if a.endswith(".py") and not a.startswith("-")]
    sb.restore_tests(extras)
    ans = sb.read_answer()
    want = answer_marker(instruction)
    if ans.strip().upper() not in (want, "DONE"):
        return False, f"answer.txt must contain {want}, got {ans[:80]!r}"
    if not cmds:
        return True, "no test command found; accepted DONE"
    for argv in cmds:
        rc, out = sb.run_command(argv, timeout=20)
        if rc != 0:
            return False, f"tests failed: {' '.join(argv)}\n{out[-1200:]}"
    return True, "tests pass"


def solve_llm(task, client: LLMClient, meter: TaskMeter, model: str, max_tokens: int = 0):
    """Returns (answer|None, source).

    Attempt 1 is a compact JSON plan for math/data/search (executed locally by
    harness.dsl), else a script. Attempt 2 is always a full script, with the
    plan error as feedback. Total: at most 2 LLM calls.
    """
    max_tokens = max_tokens or MAX_OUT.get(task.get("type"), 600)  # tight cap per type: cheaper reservations
    mult = out_mult(model)
    max_tokens = max(1, int(max_tokens * mult))
    coding = task.get("type") in CODING_TYPES
    marker = answer_marker(task["instruction"]) if coding else ""
    ans_path = os.path.join(task["dir"], "answer.txt")
    try:
        os.remove(ans_path)  # never let a stale answer make an unsolved task look solved
    except OSError:
        pass

    spec = None if coding or os.environ.get("DISABLE_DSL") == "1" else dsl.spec(task)
    patch_spec = patch.spec(task) if coding and os.environ.get("DISABLE_PATCH") != "1" else None
    script_messages = [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": task["instruction"] + "\n\n" + build_context(task["dir"])}]
    sb = runner.Sandbox(task["dir"])
    try:
        for attempt in range(2):
            if attempt == 0 and patch_spec:
                system, ctx = patch_spec
                content = task["instruction"]
                if ctx:
                    content += "\n\n" + ctx
                try:
                    reply = client.chat([{"role": "system", "content": system},
                                         {"role": "user", "content": content}],
                                        model, max_tokens, meter, tag=f"{task['id']}#patch")
                except BudgetExceeded:
                    return None, "budget"
                except RuntimeError:
                    return None, "llm_error"
                edits, err = patch.parse(reply.text)
                if edits is not None:
                    originals = None
                    try:
                        originals = patch.apply(sb.dir, edits)
                        with open(os.path.join(sb.dir, "answer.txt"), "w", encoding="utf-8") as f:
                            f.write(marker)
                        ok, detail = _verify_coding(sb, task["instruction"])
                    except patch.PatchError as e:
                        ok, detail = False, f"replacement failed: {e}"
                    if ok:
                        sb.commit(("answer.txt",) + tuple(sb.changed_python()))
                        return marker, "patch0"
                    if originals:
                        patch.revert(sb.dir, originals)
                    script_messages += [
                        {"role": "assistant", "content": reply.text},
                        {"role": "user", "content": f"The replacement plan did not pass: {detail}\n"
                                                    "Reply with the full python3 script instead."}]
                    continue
                code = extract_code(reply.text)  # the prompt allows a script when a patch is not enough
                if code:
                    rc, out = sb.run(code)
                    ok, detail = _verify_coding(sb, task["instruction"])
                    if ok:
                        sb.commit(("answer.txt",) + tuple(sb.changed_python()))
                        return marker, "llm0"
                    fb = (f"Script exit {rc}. {detail}\n"
                          "Fix the code (do not edit the tests) and reply with the full script.")
                else:
                    fb = (f"The replacement plan was rejected: {err}\n"
                          "Reply with the full python3 script instead.")
                script_messages += [{"role": "assistant", "content": reply.text},
                                    {"role": "user", "content": fb}]
                continue
            if attempt == 0 and spec:
                content = task["instruction"]
                if spec.context:
                    content += "\n\n" + spec.context
                messages = [{"role": "system", "content": spec.system},
                            {"role": "user", "content": content}]
                dsl_cap = spec.max_tokens if mult == 1 else max(1200, int(spec.max_tokens * mult))
                try:
                    reply = client.chat(messages, model, dsl_cap, meter, tag=f"{task['id']}#dsl")
                except BudgetExceeded:
                    return None, "budget"
                except RuntimeError:
                    return None, "llm_error"
                ans, err = dsl.execute(task, reply.text)
                if ans is not None:
                    return ans, "dsl0"
                script_messages += [
                    {"role": "assistant", "content": reply.text},
                    {"role": "user", "content": f"The JSON plan failed: {err}\n"
                                                "Write the full python3 script instead."}]
                continue
            try:
                reply = client.chat(script_messages, model, max_tokens, meter, tag=f"{task['id']}#{attempt}")
            except BudgetExceeded:
                return None, "budget"
            except RuntimeError:
                return None, "llm_error"
            code = extract_code(reply.text)
            if not code:
                fb = "No ```python block found. Reply with one script."
            else:
                rc, out = sb.run(code)
                if coding:
                    ok, detail = _verify_coding(sb, task["instruction"])
                    if ok:
                        sb.commit(("answer.txt",) + tuple(sb.changed_python()))
                        return marker, f"llm{attempt}"
                    fb = (f"Script exit {rc}. {detail}\n"
                          "Fix the code (do not edit the tests) and reply with the full script.")
                else:
                    ans = sb.read_answer()
                    if rc == 0 and ans:
                        sb.commit(("answer.txt",))
                        return ans, f"llm{attempt}"
                    fb = (f"Exit {rc}. Output:\n{out[-500:]}\n"
                          f"answer.txt {'empty' if not ans else 'ok'}. Fix and reply with the full script.")
            script_messages += [{"role": "assistant", "content": reply.text}, {"role": "user", "content": fb}]
        return None, "failed"
    finally:
        sb.cleanup()
