"""Logic-puzzle consistency check.

The solver handles a small but useful set of deterministic consistency checks:
- total count plus disjoint parts must match;
- explicit "the rest" or "remaining" must close the total;
- contradictory partition claims are provably unsatisfiable.
Everything else returns None so the LLM can handle the truly ambiguous cases.
"""
import re

from . import UNSOLVABLE


STOP_WORDS = {
    "are", "is", "there", "the", "a", "an", "and", "or", "of", "in",
    "to", "for", "with", "by", "on", "at", "it", "all", "exactly", "sum",
    "total", "contains", "contain", "holds", "hold", "rest", "remaining",
    "leftover", "other", "others", "else", "than", "both", "more"
}


def _norm_noun(word: str):
    w = word.lower().strip()
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
        return w[:-1]
    return w


def _match_target_noun(word: str, target: str):
    if not word:
        return False
    w = _norm_noun(word)
    t = _norm_noun(target)
    return w == t or w in {t, t + "s"} or t in {w, w + "s"}


def _extract_total(low: str):
    total_m = re.search(
        r"\b(?:exactly|contains?|holds?|there\s+are|there\s+is|total\s+of|sum\s+of|a\s+total\s+of)\s+(\d+)\s+([a-z]+(?:\s+[a-z]+)?)",
        low,
    )
    if total_m:
        return int(total_m.group(1)), total_m.group(2).split()[-1], total_m.span()
    return None


def _extract_parts(low: str, target_noun: str):
    found = {}
    total_match = re.search(
        r"\b(?:exactly|contains?|holds?|there\s+are|there\s+is|total\s+of|sum\s+of|a\s+total\s+of)\s+(\d+)\s+([a-z]+(?:\s+[a-z]+)?)",
        low,
    )
    total_span = total_match.span() if total_match else None

    for m in re.finditer(r"(\d+)\s+([a-z]+(?:\s+[a-z]+)?)", low):
        if total_span and m.span()[0] >= total_span[0] and m.span()[1] <= total_span[1]:
            continue
        number = int(m.group(1))
        words = m.group(2).split()
        if not words:
            continue
        final_word = words[-1].strip(".,;:!?()[]{}")
        if final_word in STOP_WORDS:
            continue
        # Keep explicit category counts regardless of the exact total noun; the partition checks
        # below decide whether they are consistent with the total.
        if _match_target_noun(final_word, target_noun) or final_word not in STOP_WORDS:
            found[m.start()] = number

    # explicit "the rest are X" / "remaining X" / "leftover X"
    rem = re.search(r"\b(?:rest|remaining|leftover)\b[^\n.!?]*?(?:is|are)\s*(\d+)\b", low)
    if rem:
        found[10**9] = int(rem.group(1))

    return list(found.values())


def solve(instruction: str, task_dir: str):
    low = instruction.lower()
    total_info = _extract_total(low)
    if not total_info:
        return None
    total, noun, _ = total_info

    parts = _extract_parts(low, noun)
    if not parts:
        return None

    # A partition that exceeds the total is immediately impossible.
    if sum(parts) > total:
        return UNSOLVABLE

    explicit_partition = bool(re.search(
        r"\bno\s+other\b|\bno\s+others\b|\bnothing\s+else\b|\bonly\s+(?:these|those|the\s+following)\b|"
        r"\bconsists?\s+only\s+of\b|\b(?:rest|remaining|leftover)\b",
        low,
    ))
    if explicit_partition and sum(parts) != total:
        return UNSOLVABLE

    return None
