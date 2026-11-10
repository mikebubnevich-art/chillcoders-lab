"""LLM as a compiler: the model returns a compact JSON plan, local code runs it.

Why: a full python script costs 300-900 output tokens and can still fail on
arithmetic or answer formatting. A JSON plan costs ~60-160 output tokens and
the interpreter below is exact (Fraction for math, csv for data, KB graph for
search). If a plan cannot be parsed/validated/executed we return a short error
and the agent falls back to the full-script mode, so the total stays 2 calls.

Safety rule: a plan that does not *exactly* describe the computation must be
rejected, never guessed. `_run_math` refuses non-exact division unless the plan
says round down/up; `_run_data` refuses unknown columns/keys and non-integral
answers without an explicit decimals hint; `_run_search` requires an exact
entity title and an unambiguous field at every hop.
"""
import csv
import json
import os
import re
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction

from solvers import search_solver

MAX_PLAN_CHARS = 4000
MAX_FILTERS = 20

SYSTEM = {
    "math": (
        "You compile a word problem into a JSON plan. Reply with ONE ```json block, no prose.\n"
        'Schema: {"start": <int>, "ops": [<op>, ...]} with at most 20 ops and no other keys.\n'
        'Add/subtract: {"op":"add"|"sub","amount":<int>,"times":<int>} (times optional, default 1) '
        'or {"op":"add"|"sub","lots":<int>,"amount":<int>} for "N deliveries/shipments of M each".\n'
        'Multiply: {"op":"mul","k":<number>}. Divide: {"op":"div","k":<number>,"round":"down"|"up"|"exact"}.\n'
        "Use \"exact\" only when the division is even; use down/up when the text says rounded down/up.\n"
        "Local code does the arithmetic with exact fractions.\n"
        'Example: {"start":1669,"ops":[{"op":"sub","amount":36,"times":3},'
        '{"op":"add","lots":2,"amount":87}]}'
    ),
    "data_analysis": (
        "You compile a data question into a JSON plan. Reply with ONE ```json block, no prose.\n"
        'Schema: {"filters":[{"col":<name>,"op":<op>,...}, ...], "agg":<agg>, "col":<name>?,'
        ' "group":<name>?, "mode":<mode>?, "direction":"max"|"min"?, "decimals":<int>?}\n'
        "Filter ops: eq, ne, gt, ge, lt, le, between (lo,hi), in (values), not_in (values), "
        "contains, startswith. All filters are ANDed.\n"
        "Aggs: sum, count, count_distinct, avg, median, min, max, group_top, percent, "
        "sum_product.\n"
        "percent = percentage of the filtered rows among all rows; it needs decimals "
        "(e.g. 'What percentage of orders ...' -> agg=percent).\n"
        "sum_product needs cols:[<col1>,<col2>] and returns sum(col1*col2) per row "
        "(e.g. total revenue -> cols:[\"units\",\"price\"]).\n"
        "For group_top set group, col (omit for mode=count), mode=sum|avg|count|min|max, "
        "direction=max|min; the answer is the group name and ties are rejected.\n"
        'Example: {"filters":[{"col":"product","op":"eq","value":"widget"},'
        '{"col":"month","op":"between","lo":7,"hi":8}],"agg":"sum","col":"units"}'
    ),
    "search": (
        "You compile a knowledge-base question into a JSON plan. Reply with ONE ```json block, "
        'no prose. Schema: {"start":<entity title>,"hops":[<field>, ...]} with no other keys.\n'
        "Fields are followed in order; the value of the last field is the answer. "
        "Use the field names from the KB schema, or a common synonym (city, works at, founded by).\n"
        'Example: {"start":"Alice Smith","hops":["employer","headquarters"]}'
    ),
}
MAX_TOKENS = {"math": 130, "data_analysis": 200, "search": 100}

JSON_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.S)
CSV_RE = re.compile(r"([\w.\-]+\.csv)")
DEC_RE = re.compile(r"round(?:ed)?\s+to\s+(\d+)\s+decimal", re.I)
DEC2_RE = re.compile(r"(\d+)\s+decimal\s+places?", re.I)


class PlanError(Exception):
    pass


class Spec:
    def __init__(self, system: str, max_tokens: int, context: str):
        self.system = system
        self.max_tokens = max_tokens
        self.context = context


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _is_int(v) -> bool:
    return not isinstance(v, bool) and isinstance(v, int)


def _int(v, name: str) -> int:
    if _is_int(v):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    raise PlanError(f"{name} must be an integer, got {v!r}")


def _positive(v, name: str) -> Fraction:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise PlanError(f"{name} must be a number, got {v!r}")
    try:
        f = Fraction(str(v))
    except (ValueError, ZeroDivisionError):
        raise PlanError(f"{name} must be a number, got {v!r}")
    if f <= 0:
        raise PlanError(f"{name} must be positive")
    return f


def _keys(obj, allowed, required=()):
    extra = set(obj) - set(allowed)
    if extra:
        raise PlanError(f"unknown keys {sorted(extra)}; allowed: {sorted(allowed)}")
    missing = [k for k in required if k not in obj]
    if missing:
        raise PlanError(f"missing keys {missing}")


def _csv_schema(task) -> str:
    text = task.get("instruction", "")
    m = CSV_RE.search(text)
    names = [m.group(1)] if m else []
    if not names:
        d = task["dir"]
        names = sorted(fn for fn in os.listdir(d) if fn.lower().endswith(".csv"))[:1]
    if not names:
        return ""
    name = names[0]
    path = os.path.join(task["dir"], name)
    try:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            rows = []
            for i, row in enumerate(csv.DictReader(f)):
                rows.append(row)
                if i >= 199:
                    break
    except OSError:
        return ""
    if not rows:
        return f"CSV {name}: empty."
    cols = list(rows[0].keys())
    numeric = [c for c in cols if all(r.get(c) not in (None, "") and _num(r[c]) is not None for r in rows)]
    return f"CSV {name} columns: {','.join(cols)}. Numeric: {','.join(numeric) or 'none'}."


def _kb_schema(task) -> str:
    kb = os.path.join(task["dir"], "kb")
    if not os.path.isdir(kb):
        return ""
    try:
        ents, _ = search_solver._load(kb)
    except OSError:
        return ""
    if not ents:
        return ""
    fields = sorted(search_solver._field_keys(ents))
    return (f"KB has {len(ents)} entities. Fields: {','.join(fields[:40])}"
            + (" ..." if len(fields) > 40 else ""))


def context(task) -> str:
    # A schema bug must only make the plan less informed, never kill the task
    # before the LLM attempt; the script fallback still has the full context.
    try:
        t = task.get("type")
        if t == "data_analysis":
            return _csv_schema(task)
        if t == "search":
            return _kb_schema(task)
        if t == "math":
            return _file_preview(task)
        return ""
    except Exception:
        return ""


DATA_FILE_RE = re.compile(r"(?<![\w/.:-])((?:[\w.-]+/)*[\w.-]+\.(?:csv|json|tsv|txt))\b")


def _file_preview(task, limit=1200) -> str:
    """Show the small data files a maths plan may need to read a starting value from.

    Without this the plan schema (literal start + literal ops) cannot express
    "starts with the stock listed in inventory.csv", and models silently guess 0.
    """
    instr = task.get("instruction", "")
    chunks = []
    for name in dict.fromkeys(DATA_FILE_RE.findall(instr)):
        path = os.path.join(task.get("dir", ""), name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read(limit)
        except OSError:
            continue
        chunks.append(f"file {name}:\n{text.strip()}")
    return "\n\n".join(chunks)


def spec(task):
    t = task.get("type")
    if t not in SYSTEM:
        return None
    return Spec(SYSTEM[t], MAX_TOKENS[t], context(task))


def _parse(text: str) -> dict:
    if len(text) > MAX_PLAN_CHARS:
        raise PlanError("reply too long for a plan")
    m = JSON_RE.search(text)
    raw = (m.group(1) if m else text).strip()
    if not raw:
        raise PlanError("empty reply")
    try:
        obj = json.loads(raw)
    except ValueError:
        i, j = raw.find("{"), raw.rfind("}")
        if i < 0 or j <= i:
            raise PlanError("no JSON object in reply")
        try:
            obj = json.loads(raw[i:j + 1])
        except ValueError as e:
            raise PlanError(f"invalid JSON: {e}")
    if not isinstance(obj, dict):
        raise PlanError("plan must be a JSON object")
    return obj


def _fmt(v: float, decimals):
    if decimals is not None:
        return _half_up(repr(v), decimals)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    raise PlanError("non-integer answer without a decimals hint; cannot format")


def _half_up(value, decimals: int) -> str:
    """Round half away from zero on the printed decimal, not half-to-even on the binary float."""
    q = Decimal(1).scaleb(-decimals)
    return str(Decimal(value).quantize(q, rounding=ROUND_HALF_UP))


def _decimals(instruction: str, plan: dict):
    m = DEC_RE.search(instruction) or DEC2_RE.search(instruction)
    if m:
        return int(m.group(1))
    v = plan.get("decimals")
    if v is None:
        return None
    n = _int(v, "decimals")
    if not 0 <= n <= 6:
        raise PlanError("decimals must be between 0 and 6")
    return n


# ---------------- math ----------------

def _fmt_fraction(n: Fraction, instruction: str, plan: dict) -> str:
    decimals = _decimals(instruction, plan)
    if decimals is not None:
        return _half_up(Decimal(n.numerator) / Decimal(n.denominator), decimals)
    if n.denominator == 1:
        return str(n.numerator)
    sign = "-" if n < 0 else ""
    n = abs(n)
    d, twos, fives = n.denominator, 0, 0
    while d % 2 == 0:
        d //= 2
        twos += 1
    while d % 5 == 0:
        d //= 5
        fives += 1
    if d != 1:
        raise PlanError("non-terminating decimal; add a decimals hint or round down/up")
    places = max(twos, fives)
    scaled = n * 10 ** places
    if scaled.denominator != 1:
        raise PlanError("internal formatting error")
    digits = str(scaled.numerator).rjust(places + 1, "0")
    out = (digits[:-places] + "." + digits[-places:]).rstrip("0").rstrip(".") if places else digits
    return sign + out


def _run_math(task, plan):
    _keys(plan, {"start", "ops", "decimals"}, required=("start", "ops"))
    n = Fraction(_int(plan["start"], "start"))
    instruction = task.get("instruction", "")
    if n == 0:
        # A literal 0 is the classic silent guess when the real start value lives
        # in a file the plan cannot read: refuse instead of returning a wrong number.
        files = [f for f in dict.fromkeys(DATA_FILE_RE.findall(instruction))
                 if os.path.isfile(os.path.join(task.get("dir", ""), f))]
        if files:
            raise PlanError("start is 0 but the instruction reads the starting value from "
                            + ", ".join(files))
    ops = plan["ops"]
    if not isinstance(ops, list) or not ops:
        raise PlanError("ops must be a non-empty list")
    if len(ops) > 20:
        raise PlanError("too many ops")
    for op in ops:
        if not isinstance(op, dict) or not isinstance(op.get("op"), str):
            raise PlanError("every op must be an object with an op name")
        kind = op["op"]
        if kind in ("add", "sub"):
            _keys(op, {"op", "amount", "times", "lots"}, required=("op", "amount"))
            if "times" in op and "lots" in op:
                raise PlanError("use either times or lots, not both")
            amount = _int(op["amount"], "amount")
            mult = _int(op.get("times", op.get("lots", 1)), "times/lots")
            if amount < 0 or mult < 1:
                raise PlanError("amount must be >= 0 and times/lots >= 1")
            delta = Fraction(amount) * mult
            n += delta if kind == "add" else -delta
        elif kind == "mul":
            _keys(op, {"op", "k"}, required=("op", "k"))
            n *= _positive(op["k"], "k")
        elif kind == "div":
            _keys(op, {"op", "k", "round"}, required=("op", "k"))
            k = _positive(op["k"], "k")
            rnd = op.get("round", "exact")
            if rnd not in ("down", "up", "exact"):
                raise PlanError('round must be "down", "up" or "exact"')
            q = n / k
            if rnd == "down":
                n = Fraction(q.numerator // q.denominator)
            elif rnd == "up":
                n = Fraction(-((-q.numerator) // q.denominator))
            else:
                n = q
        else:
            raise PlanError(f"unknown op {kind!r}")
    return _fmt_fraction(n, task.get("instruction", ""), plan)


# ---------------- data ----------------

def _col(name, cols):
    if not isinstance(name, str):
        raise PlanError(f"column name must be a string, got {name!r}")
    hits = [c for c in cols if c.lower() == name.strip().lower()]
    if len(hits) != 1:
        raise PlanError(f"unknown column {name!r}; columns are {cols}")
    return hits[0]


def _values(rows, col):
    return [r.get(col) for r in rows if r.get(col) not in (None, "")]


def _numeric_values(rows, col, what):
    vals = []
    for r in rows:
        raw = r.get(col)
        if raw in (None, ""):
            continue
        v = _num(raw)
        if v is None:
            raise PlanError(f"{what} {col!r} has a non-numeric value {raw!r}")
        vals.append(v)
    return vals


def _filter(rows, f, cols):
    if not isinstance(f, dict):
        raise PlanError("every filter must be an object")
    op = f.get("op")
    if not isinstance(op, str):
        raise PlanError("filter op is missing")
    allowed = {"col", "op", "value", "values", "lo", "hi"}
    _keys(f, allowed, required=("op", "col"))
    col = _col(f["col"], cols)
    if op in ("eq", "ne", "gt", "ge", "lt", "le", "contains", "startswith"):
        if "value" not in f:
            raise PlanError(f"{op} needs a value")
    elif op == "between":
        if "lo" not in f or "hi" not in f:
            raise PlanError("between needs lo and hi")
    elif op in ("in", "not_in"):
        if not isinstance(f.get("values"), list) or not f["values"]:
            raise PlanError("in/not_in need a non-empty values list")
    else:
        raise PlanError(f"unknown filter op {op!r}")

    def cmp(raw, value):
        a, b = _num(raw), _num(value) if not isinstance(value, bool) else None
        if a is not None and b is not None:
            return "num", a, b
        return "str", str(raw).strip().lower(), str(value).strip().lower()

    def test(raw):
        if raw in (None, ""):
            return False
        if op in ("eq", "ne"):
            kind, a, b = cmp(raw, f["value"])
            return (a == b) if op == "eq" else (a != b)
        if op in ("gt", "ge", "lt", "le"):
            kind, a, b = cmp(raw, f["value"])
            return {"gt": a > b, "ge": a >= b, "lt": a < b, "le": a <= b}[op]
        if op == "between":
            kind, a, lo = cmp(raw, f["lo"])
            kind2, _, hi = cmp(raw, f["hi"])
            if kind != kind2:
                raise PlanError("between bounds have different types")
            return lo <= a <= hi
        if op in ("in", "not_in"):
            hit = any(cmp(raw, v)[1] == cmp(raw, v)[2] for v in f["values"])
            return hit if op == "in" else not hit
        kind, a, b = cmp(raw, f["value"])
        if kind != "str":
            raise PlanError(f"{op} needs a text column")
        return b in a if op == "contains" else a.startswith(b)

    return [r for r in rows if test(r.get(col))]


def _run_data(task, plan):
    _keys(plan, {"filters", "agg", "col", "cols", "group", "mode", "direction", "decimals"},
          required=("agg",))
    agg = plan["agg"]
    if not isinstance(agg, str):
        raise PlanError("agg must be a string")
    instruction = task.get("instruction", "")
    m = CSV_RE.search(instruction)
    if not m:
        raise PlanError("no .csv file mentioned in the instruction")
    name = m.group(1)
    path = os.path.join(task["dir"], name)
    if not os.path.isfile(path):
        raise PlanError(f"{name} not found")
    try:
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        raise PlanError(f"cannot read {name}: {e}")
    if not rows:
        raise PlanError(f"{name} is empty")
    cols = list(rows[0].keys())
    filters = plan.get("filters", [])
    if not isinstance(filters, list) or len(filters) > MAX_FILTERS:
        raise PlanError("filters must be a list of at most 20 objects")
    sel = rows
    for f in filters:
        sel = _filter(sel, f, cols)
    decimals = _decimals(instruction, plan)

    if agg == "group_top":
        _keys(plan, {"filters", "agg", "col", "cols", "group", "mode", "direction", "decimals"},
              required=("agg", "group"))
        group = _col(plan["group"], cols)
        mode = plan.get("mode", "count")
        direction = plan.get("direction", "max")
        if mode not in ("sum", "avg", "count", "min", "max"):
            raise PlanError(f"unknown group mode {mode!r}")
        if direction not in ("max", "min"):
            raise PlanError('direction must be "max" or "min"')
        buckets = {}
        if mode == "count":
            for r in sel:
                buckets.setdefault(r.get(group), []).append(1.0)
        else:
            if "col" not in plan:
                raise PlanError("group_top needs a col unless mode=count")
            col = _col(plan["col"], cols)
            for r in sel:
                v = _num(r.get(col))
                if v is not None:
                    buckets.setdefault(r.get(group), []).append(v)
        if not buckets:
            raise PlanError("no rows match the filters")
        if mode == "count":
            scored = {k: float(len(v)) for k, v in buckets.items()}
        elif mode == "sum":
            scored = {k: sum(v) for k, v in buckets.items()}
        elif mode == "avg":
            scored = {k: sum(v) / len(v) for k, v in buckets.items()}
        elif mode == "min":
            scored = {k: min(v) for k, v in buckets.items()}
        else:
            scored = {k: max(v) for k, v in buckets.items()}
        best = max(scored.values()) if direction == "max" else min(scored.values())
        winners = [k for k, v in scored.items() if abs(v - best) < 1e-9]
        if len(winners) != 1:
            raise PlanError("tie between groups; cannot pick one")
        return str(winners[0])

    if agg not in ("sum", "count", "count_distinct", "avg", "median", "min", "max",
                   "percent", "sum_product"):
        raise PlanError(f"unknown agg {agg!r}")
    if agg == "percent":
        return _fmt(len(sel) * 100.0 / len(rows), decimals)
    if agg == "sum_product":
        pair = plan.get("cols")
        if not isinstance(pair, list) or len(pair) != 2:
            raise PlanError('sum_product needs "cols":[col1,col2]')
        c1, c2 = _col(pair[0], cols), _col(pair[1], cols)
        total = 0.0
        for r in sel:
            a, b = _num(r.get(c1)), _num(r.get(c2))
            if a is None or b is None:
                raise PlanError(f"non-numeric or missing value in {c1}*{c2}")
            total += a * b
        return _fmt(total, decimals)
    if agg == "count" and "col" not in plan:
        return _fmt(float(len(sel)), decimals)
    if "col" not in plan:
        raise PlanError(f"{agg} needs a col")
    col = _col(plan["col"], cols)
    if agg == "count":
        return _fmt(float(len(_values(sel, col))), decimals)
    if agg == "count_distinct":
        return str(len({str(v).strip().lower() for v in _values(sel, col)}))
    vals = _numeric_values(sel, col, "column")
    if not vals:
        if agg == "sum":
            return _fmt(0.0, decimals)
        raise PlanError("no rows match the filters")
    if agg == "sum":
        return _fmt(sum(vals), decimals)
    if agg == "avg":
        return _fmt(sum(vals) / len(vals), decimals)
    if agg == "median":
        import statistics
        return _fmt(statistics.median(vals), decimals)
    if agg == "min":
        return _fmt(min(vals), decimals)
    return _fmt(max(vals), decimals)


# ---------------- search ----------------

def _resolve_field(hop: str, fields):
    low = hop.strip().lower()
    exact = [k for k in fields if k.lower() == low]
    if exact:
        return exact[0]
    cands = [k for k in fields
             if any(low == a.lower() for a in search_solver.SYN.get(k, [k]))]
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1:
        raise PlanError(f"field {hop!r} is ambiguous: {cands}")
    return None


def _run_search(task, plan):
    _keys(plan, {"start", "hops"}, required=("start", "hops"))
    start, hops = plan["start"], plan["hops"]
    if not isinstance(start, str) or not start.strip():
        raise PlanError("start must be a non-empty string")
    if not isinstance(hops, list) or not hops or len(hops) > 8:
        raise PlanError("hops must be a list of 1..8 fields")
    kb = os.path.join(task["dir"], "kb")
    if not os.path.isdir(kb):
        raise PlanError("no kb/ directory")
    try:
        ents, _ = search_solver._load(kb)
    except OSError as e:
        raise PlanError(f"cannot read kb: {e}")
    if not ents:
        raise PlanError("knowledge base is empty")
    cur = next((t for t in ents if t.lower() == start.strip().lower()), None)
    if cur is None:
        raise PlanError(f"entity {start!r} not found")
    for i, hop in enumerate(hops):
        if not isinstance(hop, str):
            raise PlanError("every hop must be a field name string")
        key = _resolve_field(hop, ents[cur])
        if key is None:
            raise PlanError(f"field {hop!r} not found on {cur!r}")
        val = ents[cur][key]
        if i == len(hops) - 1:
            return val
        nxt = next((t for t in ents if t.lower() == val.lower()), None)
        if nxt is None:
            raise PlanError(f"hop {hop!r} value {val!r} is not a known entity")
        cur = nxt
    raise PlanError("no hops")


def execute(task, text: str):
    """(answer, None) or (None, short_error). Never raises."""
    try:
        plan = _parse(text)
        t = task.get("type")
        if t == "math":
            return _run_math(task, plan), None
        if t == "data_analysis":
            return _run_data(task, plan), None
        if t == "search":
            return _run_search(task, plan), None
        raise PlanError(f"DSL not supported for type {t!r}")
    except PlanError as e:
        return None, str(e)[:300]
    except Exception as e:  # a planner bug must fall back to the script, never kill the task
        return None, f"{type(e).__name__}: {e}"[:300]
