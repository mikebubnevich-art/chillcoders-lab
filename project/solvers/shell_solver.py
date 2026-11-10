"""Simple file-inspection tasks over integer lists or single-column data files.

This solver intentionally stays narrow: it only trusts a single text file with
one integer per line, or a single-column CSV/TSV-like file. Anything else is
left to the LLM. The goal is to cover the common low-cost deterministic cases
without guessing.
"""
import os
import re
import statistics

from . import UNSOLVABLE

INT_FILE = re.compile(r"(?<![\w/.:-])((?:[\w.-]+/)*[\w.-]+\.(?:txt|csv|log|dat|nums))\b")
NUMBER = re.compile(r"-?\d+")


def _fmt(v):
    if isinstance(v, float) and not v.is_integer():
        s = format(v, ".10g")
        if "." in s or "e" in s.lower():
            return s.rstrip("0").rstrip(".")
        return s
    return str(int(v))


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


def _filtered_values(vals, low):
    subset = list(vals)

    parity = None
    if "even" in low and "odd" not in low:
        parity = 0
    elif "odd" in low and "even" not in low:
        parity = 1
    if parity is not None:
        subset = [v for v in subset if v % 2 == parity]

    def add_threshold(predicate):
        nonlocal subset
        subset = [v for v in subset if predicate(v)]

    for pattern, predicate in [
        (r"\b(?:greater than|more than|above|larger than|bigger than)\s*(-?\d+)", lambda v, n: v > n),
        (r"\b(?:less than|under|below)\s*(-?\d+)", lambda v, n: v < n),
        (r"\b(?:at least|min(?:imum)?|no less than)\s*(-?\d+)", lambda v, n: v >= n),
        (r"\b(?:at most|max(?:imum)?|no more than)\s*(-?\d+)", lambda v, n: v <= n),
    ]:
        m = re.search(pattern, low)
        if m:
            n = int(m.group(1))
            add_threshold(lambda v, n=n: predicate(v, n))

    for pattern, predicate in [
        (r"\bbetween\s*(-?\d+)\s+and\s+(-?\d+)\b", lambda v, a, b: a <= v <= b),
        (r"\bfrom\s*(-?\d+)\s+to\s+(-?\d+)\b", lambda v, a, b: a <= v <= b),
    ]:
        m = re.search(pattern, low)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > b:
                a, b = b, a
            add_threshold(lambda v, a=a, b=b: predicate(v, a, b))

    return subset


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
    hit = _filtered_values(vals, low)
    if not hit:
        # For tasks asking about a non-existent subset (e.g. even numbers in a list of odds),
        # prefer an explicit empty result instead of pretending the whole file was used.
        if any(token in low for token in ("even", "odd", "greater than", "less than", "at least", "at most", "between", "from ")):
            if re.search(r"\b(?:sum|total)\b", low):
                return "0"
            if re.search(r"\b(?:count|number of|how many)\b", low):
                return "0"
        return None

    if re.search(r"\b(?:largest|greatest|maximum|max|highest)\b.*\b(?:even|odd)\b", low) or \
            re.search(r"\b(?:even|odd)\b.*\b(?:largest|greatest|maximum|max|highest)\b", low):
        return str(max(hit))
    if re.search(r"\b(?:smallest|least|minimum|min|lowest)\b.*\b(?:even|odd)\b", low) or \
            re.search(r"\b(?:even|odd)\b.*\b(?:smallest|least|minimum|min|lowest)\b", low):
        return str(min(hit))
    if re.search(r"\b(?:sum|total)\s+(?:of\s+)?(?:the\s+)?(?:even|odd)\b", low) or \
            re.search(r"\b(?:sum|total)\b.*\b(?:even|odd)\b", low):
        return _fmt(sum(hit))
    if re.search(r"\b(?:count|number of)\s+(?:the\s+)?(?:even|odd)\b", low) or \
            re.search(r"\b(?:count|number of)\b.*\b(?:even|odd)\b", low):
        return str(len(hit))
    if re.search(r"\baverage\b|\bmean\b", low):
        return _fmt(sum(hit) / len(hit))
    if re.search(r"\bmedian\b", low):
        return _fmt(statistics.median(hit))
    if re.search(r"\bsum\b|\btotal\b", low):
        return _fmt(sum(hit))
    if re.search(r"\b(?:largest|greatest|maximum|max|highest)\b", low):
        return str(max(hit))
    if re.search(r"\b(?:smallest|least|minimum|min|lowest)\b", low):
        return str(min(hit))
    if re.search(r"\bhow many\b|\bcount\b|\bnumber of\b", low):
        return str(len(hit))
    if re.search(r"\b(first|last)\b.*\b(?:value|number)\b", low):
        return str(hit[0] if "first" in low else hit[-1])
    return None
