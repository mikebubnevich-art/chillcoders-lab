"""0-token feasibility heuristics. 'hard' => skip the task without spending anything.

Rules (cheap, local):
  - instruction references a file/dir that does not exist in the task dir
  - instruction needs the internet / an external service (we only have the LLM proxy)
  - empty / too short instruction
Soft signals lower p_feasible, used by the planner.
"""
import os
import re
from dataclasses import dataclass, field

FILE_RE = re.compile(r"(?<![\w/.:-])((?:[\w.-]+/)*[\w.-]+\.(?:csv|json|py|md|txt|xlsx|tsv|log|yaml|yml))\b")
IGNORE = {"answer.txt", "task.json"}
NET_HARD = re.compile(r"https?://|\b(?:from the internet|search the web|web search|download|scrape|crawl|browse to)\b", re.I)
NET_SOFT = re.compile(r"\b(?:online|api key|live price|current price|latest news|today's)\b", re.I)
# "write the result to results.csv" references an OUTPUT file: it need not exist.
WRITE_CTX = re.compile(r"\b(?:write|save|output|store|put|create|produce|record|export|dump|submit|place)\b"
                       r"[^.;\n]{0,80}$", re.I)


def _is_output_ref(text: str, start: int) -> bool:
    return bool(WRITE_CTX.search(text[max(0, start - 80):start]))


@dataclass
class Feasibility:
    p: float = 0.95
    hard: bool = False
    reasons: list = field(default_factory=list)


def check(task: dict) -> Feasibility:
    r = Feasibility()
    text, d = task.get("instruction", ""), task["dir"]
    if len(text.strip()) < 30:
        return Feasibility(0.0, True, ["empty_or_short_instruction"])
    for m in FILE_RE.finditer(text):
        ref = m.group(1)
        if ref in IGNORE or ref.startswith("e.g"):
            continue
        if not os.path.exists(os.path.join(d, ref)):
            if _is_output_ref(text, m.start()):
                r.p *= 0.9
                r.reasons.append(f"output_file:{ref}")
            else:
                r.hard = True
                r.reasons.append(f"missing_file:{ref}")
    for m in re.finditer(r"\b([\w-]+)/(?=[\s.,)`'\"]|$)", text):  # directory mentions like kb/
        name = m.group(1)
        if name in ("and", "or") or os.path.isdir(os.path.join(d, name)):
            continue
        if _is_output_ref(text, m.start()):
            r.p *= 0.9
            r.reasons.append(f"output_dir:{name}/")
        else:
            r.hard = True
            r.reasons.append(f"missing_dir:{name}")
    if NET_HARD.search(text):
        r.hard = True
        r.reasons.append("needs_internet")
    elif NET_SOFT.search(text):
        r.p *= 0.3
        r.reasons.append("maybe_needs_internet")
    if r.hard:
        r.p = 0.0
    return r
