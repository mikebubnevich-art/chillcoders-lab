"""Single-CSV aggregation questions. Parses filters + aggregate, computes with stdlib csv."""
import csv
import os
import re
import statistics

from . import UNSOLVABLE


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(v: float, decimals):
    if decimals is not None:
        return f"{v:.{decimals}f}"
    return str(int(v)) if abs(v - round(v)) < 1e-9 else None  # unknown float format -> LLM


def solve(instruction: str, task_dir: str):
    m = re.search(r"([\w.\-]+\.csv)", instruction)
    if not m:
        return None
    path = os.path.join(task_dir, m.group(1))
    if not os.path.isfile(path):
        return UNSOLVABLE
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cols = list(rows[0].keys())
    qs = [x for x in re.split(r"(?<=[.?!])\s+", instruction) if x.rstrip().endswith("?")]
    if len(qs) != 1:
        return None
    full_low = instruction.lower()  # formatting hints ("round to 2 decimals") may follow the question
    instruction = qs[0]  # ONLY the question: the preamble lists all column names
    low = instruction.lower()
    numeric = [c for c in cols if all(_num(r[c]) is not None for r in rows)]
    cat = [c for c in cols if c not in numeric]

    sel = rows
    # month / numeric range filters ("months 7 through 8", "month 3", "months 2-4")
    for c in numeric:
        cn = re.escape(c.lower())
        r = re.search(rf"{cn}s?\s+(\d+)\s*(?:through|to|-|–|and)\s*(\d+)", low)
        if r and c not in ("units", "price"):
            a, b = int(r.group(1)), int(r.group(2))
            sel = [x for x in sel if a <= _num(x[c]) <= b]
            continue
        r = re.search(rf"\b(?:in|for|during)\s+{cn}\s+(\d+)\b", low)
        if r and c not in ("units", "price"):
            sel = [x for x in sel if _num(x[c]) == int(r.group(1))]
    # categorical filters: quoted values first, then bare whole-word values
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", instruction)
    used = set()
    for q in quoted:
        hit = [c for c in cat if any(x[c].lower() == q.lower() for x in rows)]
        if len(hit) != 1:
            if not hit:
                continue  # quote is not a data value (e.g. a column name)
            return None
        sel = [x for x in sel if x[hit[0]].lower() == q.lower()]
        used.add(hit[0])
    for c in cat:
        if c in used or c.lower().endswith("id"):
            continue
        vals = {x[c].lower() for x in rows}
        found = [v for v in vals if len(v) >= 3 and re.search(rf"\b{re.escape(v)}\b", low)]
        if len(found) == 1:
            sel = [x for x in sel if x[c].lower() == found[0]]

    dm = re.search(r"round(?:ed)?\s+to\s+(\d+)\s+decimal", full_low)
    decimals = int(dm.group(1)) if dm else None

    if "revenue" in low and "units" in numeric and "price" in numeric:
        return _fmt(sum(_num(x["units"]) * _num(x["price"]) for x in sel), decimals)
    if re.search(r"(how many|number of|count of)\s+(orders|rows|records|sales)", low):
        return str(len(sel))
    dm = re.search(r"(?:how many|number of|count of)\s+(?:unique|distinct)\s+([a-z_]+)", low)
    if dm:
        w = dm.group(1)
        col = next((c for c in cols if c.lower() in (w, w.rstrip("s"))), None)
        return None if col is None else str(len({x[col] for x in sel}))
    pm = re.search(r"\b(?:percentage|percent)\s+of\s+(?:the\s+)?(?:orders|rows|records|sales)\b", low)
    if pm:
        return _fmt(100.0 * len(sel) / len(rows), decimals)
    # "which product had the highest total units?", "which region sold the most orders?"
    gm = re.search(r"(?:which|what)\s+([a-z_]+)\s+(?:had|has|have|sold|generated|recorded|showed)?\s*"
                   r"(?:the\s+)?(highest|largest|most|lowest|smallest|least)\s+"
                   r"(?:total\s+|average\s+|mean\s+)?([a-z_]+)", low)
    if gm:
        gword, direction, tword = gm.groups()
        group_col = next((c for c in cols if c.lower() in (gword, gword.rstrip("s"))), None)
        if group_col:
            count_mode = tword in ("orders", "rows", "records")
            target = next((c for c in numeric if c.lower() in (tword, tword.rstrip("s")) and c != "order_id"), None)
            revenue_mode = tword in ("revenue", "sales") and "units" in numeric and "price" in numeric
            if not (count_mode or target or revenue_mode):
                return None
            buckets = {}
            for x in sel:
                if count_mode:
                    v = 1.0
                elif revenue_mode:
                    v = _num(x["units"]) * _num(x["price"])
                else:
                    v = _num(x[target])
                if v is not None:
                    buckets.setdefault(x[group_col], []).append(v)
            if not buckets:
                return None
            if count_mode:
                scored = {k: float(len(vs)) for k, vs in buckets.items()}
            elif re.search(r"\b(?:average|mean)\b", low):
                scored = {k: sum(vs) / len(vs) for k, vs in buckets.items()}
            elif target and "price" in target.lower():
                scored = {k: (max(vs) if direction in ("highest", "largest", "most") else min(vs))
                          for k, vs in buckets.items()}
            else:
                scored = {k: sum(vs) for k, vs in buckets.items()}
            best = max(scored.values()) if direction in ("highest", "largest", "most") else min(scored.values())
            winners = [k for k, v in scored.items() if abs(v - best) < 1e-9]
            return winners[0] if len(winners) == 1 else None
    head = re.search(r"(?:how many|total|sum of|average|mean|median|maximum|minimum|highest|lowest|largest|smallest)"
                     r"\s+(?:of\s+)?(?:the\s+)?([a-z_]+)", low)
    target = None
    if head:
        w = head.group(1)
        target = next((c for c in numeric if c.lower() in (w, w.rstrip("s")) and c != "order_id"), None)
    if target is None:
        return None
    vals = [_num(x[target]) for x in sel]
    if not vals:
        return None
    if re.search(r"\bmedian\b", low):
        return _fmt(statistics.median(vals), decimals)
    if re.search(r"\b(average|mean)\b", low):
        return _fmt(sum(vals) / len(vals), decimals)
    if re.search(r"\b(maximum|highest|largest|max)\b", low):
        return _fmt(max(vals), decimals)
    if re.search(r"\b(minimum|lowest|smallest|min)\b", low):
        return _fmt(min(vals), decimals)
    if re.search(r"how many|total|sum", low):
        return _fmt(sum(vals), decimals)
    return None
