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
    for ref in set(FILE_RE.findall(text)):
        if ref in IGNORE or ref.startswith("e.g"):
            continue
        if not os.path.exists(os.path.join(d, ref)):
            r.hard = True
            r.reasons.append(f"missing_file:{ref}")
    for m in re.findall(r"\b([\w-]+)/(?=[\s.,)])", text):  # directory mentions like kb/
        if m not in ("and", "or") and not os.path.isdir(os.path.join(d, m)):
            r.hard = True
            r.reasons.append(f"missing_dir:{m}")
    if NET_HARD.search(text):
        r.hard = True
        r.reasons.append("needs_internet")
    elif NET_SOFT.search(text):
        r.p *= 0.3
        r.reasons.append("maybe_needs_internet")
    if r.hard:
        r.p = 0.0
    return r
