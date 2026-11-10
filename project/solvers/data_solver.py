"""Single-CSV aggregation questions. Parses filters + aggregate, computes with stdlib csv."""
import csv
import os
import re
import statistics
from decimal import Decimal, ROUND_HALF_UP

from . import UNSOLVABLE

MONTH_IDX = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTH_FULL = {
    "january": "jan", "february": "feb", "march": "mar", "april": "apr", "may": "may",
    "june": "jun", "july": "jul", "august": "aug", "september": "sep", "sept": "sep",
    "october": "oct", "november": "nov", "december": "dec",
}


def _month(word: str):
    """'July' / 'jul' -> 'jul'; anything else -> None."""
    w = word.lower()
    return w if w in MONTH_IDX else MONTH_FULL.get(w)


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(v: float, decimals):
    if decimals is not None:
        # f-strings round the binary value half-to-even (16.25 -> "16.2"); the
        # tasks mean half-up on the decimal they printed (16.3).
        q = Decimal(1).scaleb(-decimals)
        return str(Decimal(repr(v)).quantize(q, rounding=ROUND_HALF_UP))
    return str(int(v)) if abs(v - round(v)) < 1e-9 else None  # unknown float format -> LLM


def solve(instruction: str, task_dir: str):
    m = re.search(r"([\w.\-]+\.csv)", instruction)
    if not m:
        return None
    path = os.path.join(task_dir, m.group(1))
    if not os.path.isfile(path):
        return UNSOLVABLE
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = list(reader.fieldnames or [])
    if not rows and not cols:
        return None  # not even a header: let the LLM look at the file
    # A question normally ends with '?'. Some templates omit it ("Find the
    # total number of units of pen sold in quarter 2."), so fall back to the
    # single sentence that asks for an aggregate. Several such sentences are
    # ambiguous -> let the LLM handle the task.
    sent = [x for x in re.split(r"(?<=[.?!])\s+", instruction) if x.strip()]
    qs = [x for x in sent if x.rstrip().endswith("?")]
    if len(qs) == 1:
        question = qs[0]
    elif not qs:
        agg = [x for x in sent if re.search(
            r"\b(?:total|sum|how many|number of|count|average|mean|median|maximum|minimum|"
            r"highest|lowest|largest|smallest|percentage|percent|revenue)\b", x, re.I)]
        if len(agg) != 1:
            return None
        question = agg[0]
    else:
        return None
    full_text = instruction  # the preamble/quotes may matter even though only the question is parsed
    full_low = full_text.lower()  # formatting hints ("round to 2 decimals") may follow the question
    instruction = question  # ONLY the question: the preamble lists all column names
    low = instruction.lower()
    col_lower = {c.lower() for c in cols}

    # Structural infeasibility: the file is not the one the question describes.
    stated = re.search(r"\(\s*(\d+)\s+rows?\s*\)", full_text, re.I)
    if stated and int(stated.group(1)) != len(rows):
        return UNSOLVABLE
    for w in re.findall(r"(?:^|[\s(])the\s+([a-z_][a-z0-9_]*)\s+column\b", low):
        if w not in col_lower and not re.search(rf"{w}\s*(?:column\s*)?(?:is|=|means)\b", low):
            return UNSOLVABLE
    if not rows:
        # header-only file: an average/median/summed value provably does not exist
        return UNSOLVABLE if re.search(r"\b(?:average|mean|median|sum|total)\b", low) else None

    numeric = [c for c in cols if all(_num(r[c]) is not None for r in rows)]
    cat = [c for c in cols if c not in numeric]

    sel = rows
    # month columns may hold 'Jul'/'Aug' while the question says 'July through August'
    for c in cols:
        vals = {x[c].strip().lower() for x in rows}
        if not vals or not vals <= set(MONTH_IDX):
            continue
        cn = re.escape(c.lower())
        rng = re.search(rf"{cn}s?\s+([a-z]+)\s*(?:through|to|until|and|-|–)\s*([a-z]+)", low)
        if rng and _month(rng.group(1)) and _month(rng.group(2)):
            a, b = MONTH_IDX[_month(rng.group(1))], MONTH_IDX[_month(rng.group(2))]
            lo, hi = min(a, b), max(a, b)
            sel = [x for x in sel if lo <= MONTH_IDX.get(x[c].strip().lower(), -1) <= hi]
            continue
        said = [w for w in (_month(m.group(1)) for m in
                            re.finditer(r"\b(?:in|during|for|months?|of)\s+([a-z]+)\b", low)) if w]
        if len(set(said)) == 1:
            sel = [x for x in sel if x[c].strip().lower() == said[0]]
    # threshold filters: "at least 20 units", "more than 5 orders"
    for m_ in re.finditer(r"\b(at\s+least|at\s+most|more\s+than|fewer\s+than|greater\s+than|less\s+than)\s+"
                          r"(\d+(?:\.\d+)?)\s+([a-z_]+)", low):
        word, num, name = m_.group(1), m_.group(2), m_.group(3)
        word = re.sub(r"\s+", " ", word)
        col = next((c for c in numeric if c.lower() in (name, name.rstrip("s")) and c != "order_id"), None)
        if col is None:
            continue
        v = float(num)
        keep = {"at least": lambda x: x >= v, "more than": lambda x: x > v,
                "greater than": lambda x: x > v, "at most": lambda x: x <= v,
                "fewer than": lambda x: x < v, "less than": lambda x: x < v}[word]
        sel = [x for x in sel if _num(x[col]) is not None and keep(_num(x[col]))]
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
    # categorical filters: quoted values first, then bare whole-word values.
    # Labels are compared stripped: 'widget ' and 'widget' are the same product.
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", instruction)
    used = set()
    for q in quoted:
        hit = [c for c in cat if any(x[c].strip().lower() == q.strip().lower() for x in rows)]
        if len(hit) != 1:
            if not hit:
                continue  # quote is not a data value (e.g. a column name)
            return None
        sel = [x for x in sel if x[hit[0]].strip().lower() == q.strip().lower()]
        used.add(hit[0])
    for c in cat:
        if c in used or c.lower().endswith("id"):
            continue
        vals = {x[c].strip().lower() for x in rows}
        found = [v for v in vals if len(v) >= 3 and re.search(rf"\b{re.escape(v)}\b", low)]
        if len(found) == 1:
            sel = [x for x in sel if x[c].strip().lower() == found[0]]

    dm = re.search(r"round(?:ed)?\s+to\s+(\d+)\s+decimal", full_low)
    decimals = int(dm.group(1)) if dm else None

    qty_col = next((c for c in numeric if c.lower() in ("units", "quantity", "qty")), None)
    price_col = next((c for c in numeric if c.lower() in ("price", "unit_price", "unitprice")), None)
    revenue_asked = ("revenue" in low and qty_col and price_col
                     and not re.search(r"\bnot\s+revenue\b|\bsum\s+units\b", low))
    if revenue_asked:
        return _fmt(sum(_num(x[qty_col]) * _num(x[price_col]) for x in sel), decimals)
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
            revenue_mode = tword in ("revenue", "sales") and bool(qty_col and price_col)
            if not (count_mode or target or revenue_mode):
                return None
            buckets = {}
            for x in sel:
                if count_mode:
                    v = 1.0
                elif revenue_mode:
                    v = _num(x[qty_col]) * _num(x[price_col])
                else:
                    v = _num(x[target])
                if v is not None:
                    buckets.setdefault(x[group_col].strip(), []).append(v)
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
            if len(winners) == 1:
                return winners[0]
            # A tie with no stated rule has no unique answer: reported as such,
            # never guessed. Explicit rules are honoured (they may sit in the tail).
            if re.search(r"alphabetic|lexicograph|dictionary\s+order", full_low):
                return sorted(winners)[0]
            if re.search(r"\bfirst\b.*\b(?:appears?|occurs?|listed|row)\b", full_low):
                return next(k for k in (x[group_col].strip() for x in rows) if k in winners)
            return UNSOLVABLE
    head = re.search(r"(?:how many|total|sum of|average|mean|median|maximum|minimum|highest|lowest|largest|smallest)"
                     r"\s+(?:of\s+)?(?:the\s+)?([a-z_]+)", low)
    # "total number of units of pen sold" -> measure is "units", not "number"
    m = re.search(r"(?:total|sum|number)\s+(?:number\s+)?of\s+([a-z_]+)", low)
    target = None
    if m:
        w = m.group(1)
        target = next((c for c in numeric if c.lower() in (w, w.rstrip("s")) and c != "order_id"), None)
    if target is None and head:
        w = head.group(1)
        target = next((c for c in numeric if c.lower() in (w, w.rstrip("s")) and c != "order_id"), None)
    if target is None:
        # A numeric aggregate was asked for, but the CSV has no numeric column
        # at all: the value is provably not in the files.
        if not numeric and re.search(
                r"\b(?:total|sum|average|mean|median|maximum|minimum|highest|lowest|largest|smallest)\b", low):
            return UNSOLVABLE
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
