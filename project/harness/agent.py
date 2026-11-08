"""Minimal one-shot agent.

Flow per task (cheapest first):
  (template solvers + feasibility run earlier in triage, 0 tokens)
  1. ONE LLM call: model writes a python3 script, we run it locally (0 tokens)
  2. at most ONE retry with the (truncated) error output, then give up
Why: history re-sent every step grows quadratically; one script call + one retry
is bounded by 2 calls and the script does the real work for free.
"""
import os
import re

from . import runner
from .budget import BudgetExceeded, TaskMeter
from .context import build_context
from .llm_client import LLMClient

SYSTEM = ("Solve the task by writing ONE python3 script (stdlib, pandas, sympy allowed). "
          "Reply with a single ```python block and no prose. The script runs in the "
          "task directory and must write ONLY the final answer to answer.txt.")

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = CODE_RE.search(text)
    if m:
        return m.group(1)
    return text if "answer.txt" in text else ""


def _read_answer(task) -> str:
    p = os.path.join(task["dir"], "answer.txt")
    try:
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


MAX_OUT = {"math": 300, "search": 400, "data_analysis": 500, "coding": 900}


def solve_llm(task, client: LLMClient, meter: TaskMeter, model: str, max_tokens: int = 0):
    """Returns (answer|None, source)."""
    max_tokens = max_tokens or MAX_OUT.get(task.get("type"), 600)  # tight cap per type: cheaper reservations
    ans_path = os.path.join(task["dir"], "answer.txt")
    if os.path.exists(ans_path):
        os.remove(ans_path)

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task["instruction"] + "\n\n" + build_context(task["dir"])}]
    for attempt in range(2):  # 1 call + at most 1 retry
        try:
            reply = client.chat(messages, model, max_tokens, meter, tag=f"{task['id']}#{attempt}")
        except BudgetExceeded:
            return None, "budget"
        except RuntimeError:
            return None, "llm_error"
        code = extract_code(reply.text)
        if code:
            rc, out = runner.run_script(code, task["dir"])
            ans = _read_answer(task)
            if rc == 0 and ans:
                return ans, f"llm{attempt}"
            fb = f"Exit {rc}. Output:\n{out[-500:]}\nanswer.txt {'empty' if not ans else 'ok'}. Fix and reply with the full script."
        else:
            fb = "No ```python block found. Reply with one script."
        messages += [{"role": "assistant", "content": reply.text}, {"role": "user", "content": fb}]
    return None, "failed"
