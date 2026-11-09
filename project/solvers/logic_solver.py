"""Logic-puzzle consistency check.

Only one claim is handled: a total is stated as "exactly N <noun>" and the
parts add up to a different number (e.g. 10 marbles = 4 red + 7 blue). With
"no other ..." or a plain arithmetic overflow this is provably inconsistent,
so the correct answer is UNSOLVABLE. Everything else returns None.
"""
import re

from . import UNSOLVABLE


def solve(instruction: str, task_dir: str):
    low = instruction.lower()
    total_m = (re.search(r"\b(?:exactly|contains?|holds?)\s+(\d+)\s+([a-z]+)", low)
               or re.search(r"\bthere\s+are\s+(\d+)\s+([a-z]+)", low))
    if not total_m:
        return None
    total, noun = int(total_m.group(1)), total_m.group(2)

    # Collect counts of the same noun ("4 red marbles", "7 blue marbles",
    # "4 red and 7 blue marbles"), excluding the total itself.
    found = {}
    pair = re.search(rf"(\d+)\s+([a-z]+)\s+and\s+(\d+)\s+([a-z]+)\s+{re.escape(noun)}\b", low)
    if pair:
        found[pair.start()] = int(pair.group(1))
        found[pair.start(3)] = int(pair.group(3))
    for m in re.finditer(rf"(\d+)\s+(?:[a-z]+\s+)*{re.escape(noun)}\b", low):
        if pair and pair.start() <= m.start() < pair.end():
            continue
        if m.start() == total_m.start():
            continue
        found[m.start()] = int(m.group(1))
    parts = list(found.values())
    if not parts:
        return None
    if sum(parts) > total:
        return UNSOLVABLE
    partition = re.search(r"\bno other\b|\bno others\b|\bnothing else\b|"
                          r"\bonly (?:these|those|the following)\b|\bconsists? only of\b", low)
    if partition and sum(parts) != total:
        return UNSOLVABLE
    return None
