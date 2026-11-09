"""Simple file-inspection tasks: largest/smallest even or odd number, sums.

Kept deliberately narrow: only a single referenced text file with one integer
per line is trusted; anything else returns None so the LLM handles it.
"""
import os
import re

from . import UNSOLVABLE

INT_FILE = re.compile(r"(?<![\w/.:-])((?:[\w.-]+/)*[\w.-]+\.(?:txt|csv|log|dat|nums))\b")
NUMBER = re.compile(r"-?\d+")


def _pick_file(instruction, task_dir):
    cands = [m.group(1) for m in INT_FILE.finditer(instruction)
             if m.group(1).lower() != "answer.txt"]
    found = [c for c in cands if os.path.isfile(os.path.join(task_dir, c))]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        return None
    if cands:
        return UNSOLVABLE  # the only referenced data file is missing
    return None


def _read_ints(path):
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    vals = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if not re.fullmatch(r"-?\d+", line):
            return None  # not the "one integer per line" shape we understand
        vals.append(int(line))
    return vals


def solve(instruction: str, task_dir: str):
    low = instruction.lower()
    rel = _pick_file(instruction, task_dir)
    if rel == UNSOLVABLE:
        return UNSOLVABLE
    if not rel:
        return None
    vals = _read_ints(os.path.join(task_dir, rel))
    if not vals:
        return None

    if re.search(r"\b(?:largest|greatest|maximum|max|highest)\b.*\beven\b", low) or \
            re.search(r"\beven\b.*\b(?:largest|greatest|maximum|max|highest)\b", low):
        hit = [v for v in vals if v % 2 == 0]
        return str(max(hit)) if hit else None
    if re.search(r"\b(?:smallest|least|minimum|min|lowest)\b.*\beven\b", low) or \
            re.search(r"\beven\b.*\b(?:smallest|least|minimum|min|lowest)\b", low):
        hit = [v for v in vals if v % 2 == 0]
        return str(min(hit)) if hit else None
    if re.search(r"\b(?:largest|greatest|maximum|max|highest)\b.*\bodd\b", low) or \
            re.search(r"\bodd\b.*\b(?:largest|greatest|maximum|max|highest)\b", low):
        hit = [v for v in vals if v % 2 == 1]
        return str(max(hit)) if hit else None
    if re.search(r"\b(?:smallest|least|minimum|min|lowest)\b.*\bodd\b", low) or \
            re.search(r"\bodd\b.*\b(?:smallest|least|minimum|min|lowest)\b", low):
        hit = [v for v in vals if v % 2 == 1]
        return str(min(hit)) if hit else None
    if re.search(r"\bsum\b|\btotal\b", low):
        return str(sum(vals))
    if re.search(r"\b(?:largest|greatest|maximum|max|highest)\b", low):
        return str(max(vals))
    if re.search(r"\b(?:smallest|least|minimum|min|lowest)\b", low):
        return str(min(vals))
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", low):
        return str(len(vals))
    return None
