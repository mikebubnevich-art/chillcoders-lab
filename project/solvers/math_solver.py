"""Chain-of-operations word problems. Strict: any unknown sentence in the chain -> None."""
import csv
import json
import math
import os
import re
from decimal import Decimal, ROUND_HALF_UP

START = re.compile(r"starts? with (\d+)", re.I)
HAS = re.compile(r"\b(?:has|have|contains?|holds?)\s+(\d+)\b", re.I)
HAS_LOTS = re.compile(
    r"\b(?:has|have|contains?|holds?)\s+(\d+)\s+\w+\s+(?:with|containing|of)\s+(\d+)\s+\w+\s+each\b",
    re.I)
# "divide the water equally among 4 tanks and keep one share"
DIVIDE = re.compile(
    r"\b(?:split|divid(?:e|es|ed))\s+(?:the\s+)?(?:\w+\s+){0,3}?equally\s+(?:into|among)\s+(\d+)\b", re.I)
KEEP = re.compile(r"\bkeeps?\s+(?:one|1)\s+(?:share|part|portion|warehouse|tank)?", re.I)
ROUND = re.compile(r"\brounded\s+(down|up)\b|\bround(?:ed)?\b(?:\s+\w+){0,4}?\s+(down|up)\b|"
                   r"discarding\s+any\s+remainder|incomplete\b", re.I)
# "remove one third of the water (round the amount removed down ...)";
# "ships out half of the remaining boxes" — the article is often dropped
FRACTION = re.compile(r"\b(?:one|1|a)?[\s-]*(third|half|quarter)\s+of\b", re.I)
FRACTION_VERB = re.compile(r"\b(?:remove|removes|removed|take|takes|took|sell|sells|sold|use|uses|used|"
                           r"lose|loses|lost|spend|spends|spent|discard|discards|discarded|"
                           r"ship|ships|shipped)\b", re.I)
BUNDLE = re.compile(r"\b(?:re)?packed\s+into\s+(?:bundles?|packs?|bags?|sacks?|boxes?|crates?)\s+of\s+(\d+)\b", re.I)
META = re.compile(r"\b(?:rounded|round(?:ing|ed)?|discard(?:ing|ed)?|remainder|incomplete|"
                  r"whole|exact(?:ly)?)\b", re.I)
MULT = {"doubled": 2, "tripled": 3, "quadrupled": 4}
MULT_ANY = re.compile(r"\b(double|doubles|doubled|triple|triples|tripled|"
                      r"quadruple|quadruples|quadrupled)\b", re.I)
MULT_VAL = {v: k for word, k in (("double", 2), ("triple", 3), ("quadruple", 4))
            for v in (word, word + "s", word + "d")}
ADD = r"(?:receives?|gets?|adds?|gains?|takes? in|buys?|purchases?|restocks?)"
SUB = (r"(?:ships?(?: out)?|sells?(?: off)?|lends?(?: out)?|loses?|removes?|uses?|donates?|discards?|"
       r"throws? away|gives? away|sends? out)")
V = rf"({ADD}|{SUB})"
SUBJ = re.compile(r"^(?:it|the \w+(?: \w+)?)\s+", re.I)
LEAD = re.compile(r"^(?:first|then|next|finally|after\s+that|afterwards)[,\s]+", re.I)
PER = re.compile(rf"{V} (\d+) \w+ per \w+ for (\d+) \w+")
# "2 deliveries of 87 boxes each", "4 boxes containing 12 books each"
LOTS = re.compile(rf"{V}\s+(\d+)\s+\w+\s+(?:of|with|containing)\s+(\d+)\s+\w+\s+each\b")
SIMPLE = re.compile(rf"{V}\s+(\d+)(?:\s+\w+){{0,2}}\s*$")
# "A depot receives 8 crates of flour, each holding 2500 grams." — start value
RECV_LOTS = re.compile(rf"{V}\s+(\d+)\s+\w+[\w ,]*?,?\s*each\s+(?:holding|containing|with|of)\s+(\d+)", re.I)
# start value read from a file: "starts with the number of bolts listed in inventory.csv",
# "starts with the stock given in config.json (key start_stock)"
FILE_START = re.compile(
    r"starts? with (?:the )?(?:(?:number|count|amount)\s+of\s+|stock\s+|value\s+)"
    r"(?P<item>[\w ]*?)\s*(?:listed|given)\s+in\s+(?P<file>[\w.\-]+\.(?:csv|json))", re.I)
JSON_KEY = re.compile(r"\(key\s+([\w.]+)\)", re.I)
# percent steps: "discounted by 10%", "8% sales tax is added"
PCT_DISC = re.compile(r"\bdiscount(?:ed)?\s+by\s+(\d+)\s*%", re.I)
PCT_TAX = re.compile(r"\b(\d+)\s*%\s+(?:sales\s+)?tax\b", re.I)
ROUND_TO = re.compile(r"round(?:ed)?\s+to\s+(\d+)\s+decimal", re.I)


def _dec(x):
    """Plain ints stay ints; '$1,204.50'-style values become Decimal."""
    s = str(x).replace("$", "").replace(",", "").strip()
    try:
        return int(s)
    except ValueError:
        pass
    try:
        from decimal import Decimal
        return Decimal(s)
    except Exception:
        return None


def _sign(verb: str) -> int:
    return 1 if re.fullmatch(ADD, verb) else -1


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def _clauses(sentence):
    """Split comma/semicolon chains: '..., then sells 59, and finally receives ...'."""
    parts = re.split(r";|,\s*(?:then|and\s+then|and\s+finally|finally|next|after\s+that|afterwards)\b",
                     sentence, flags=re.I)
    return [p.strip(" ,.") for p in parts if p.strip(" ,.")]


def _round_hint(text, default=None):
    m = ROUND.search(text)
    if not m:
        return default
    word = (m.group(1) or m.group(2) or "").lower()
    return "down" if word == "down" or "discard" in m.group(0).lower() \
        or "incomplete" in m.group(0).lower() else "up"


def _file_start(sent, task_dir):
    """'starts with the number of bolts listed in inventory.csv' /
    'starts with the stock given in config.json (key start_stock)'."""
    m = FILE_START.search(sent)
    if not m:
        return None
    path = os.path.join(task_dir, m.group("file"))
    if not os.path.isfile(path):
        return None
    if m.group("file").lower().endswith(".json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        km = JSON_KEY.search(sent)
        if not km or km.group(1) not in data:
            return None
        return _dec(data[km.group(1)])
    item = m.group("item").strip().lower()
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return None
    for row in rows:
        vals = [v for v in row.values() if v is not None]
        if not any(v.strip().lower() == item for v in vals):
            continue
        for v in vals:
            if v.strip().lower() == item:
                continue
            d = _dec(v)
            if d is not None:
                return d
    return None


COST = re.compile(r"\b(?:costs?|is\s+priced\s+at|priced\s+at)\s+\$?(\d+(?:\.\d+)?)\b", re.I)


def _initial(sent, task_dir=None):
    m = HAS_LOTS.search(sent)
    if m:
        return _dec(m.group(1)) * _dec(m.group(2))
    m = START.search(sent)
    if m:
        return _dec(m.group(1))
    m = HAS.search(sent)
    if m:
        return _dec(m.group(1))
    m = COST.search(sent)
    if m:
        return _dec(m.group(1))
    m = RECV_LOTS.search(sent)
    if m and not START.search(sent):
        # V is a capture group: 1=verb, 2=count, 3=per-unit amount
        a, b = _dec(m.group(2)), _dec(m.group(3))
        if a is not None and b is not None:
            return a * b
    if task_dir is not None:
        return _file_start(sent, task_dir)
    return None


def _apply(n, clause):
    """One chain step; returns the new value or None when the step is unknown."""
    low = clause.lower()
    m = PCT_DISC.search(clause)
    if m:
        return _pct(n, Decimal(100) - int(m.group(1)))
    m = PCT_TAX.search(clause)
    if m:
        return _pct(n, Decimal(100) + int(m.group(1)))
    if "merging" in low:
        w = next((k for k in MULT if k in low), None)
        if w:
            return n * MULT[w]
    mm = MULT_ANY.search(clause)
    if mm and not re.search(r"\b(?:every|each|per)\b", clause, re.I):
        return n * MULT_VAL[mm.group(1).lower()]

    m = DIVIDE.search(clause)
    if m:
        if not KEEP.search(clause):
            return None
        k = int(m.group(1))
        if k <= 0:
            return None
        hint = _round_hint(clause)
        if hint == "up":
            return math.ceil(n / k)
        if hint == "down":
            return n // k
        if n % k == 0:
            return n // k
        return None  # unknown rounding: never guess

    m = FRACTION.search(clause)
    if m:
        if not FRACTION_VERB.search(clause):
            return None
        word = m.group(1).lower()
        num, den = {"third": (1, 3), "half": (1, 2), "quarter": (1, 4)}[word]
        if n < 0:
            return None
        hint = _round_hint(clause)
        if hint == "down":
            removed = (n * num) // den
        elif hint == "up":
            removed = math.ceil(n * num / den)
        elif (n * num) % den == 0:
            removed = n * num // den
        else:
            return None
        return n - removed

    m = BUNDLE.search(clause)
    if m:
        k = int(m.group(1))
        return n // k if k > 0 else None

    body = LEAD.sub("", clause.strip()).rstrip(".").strip()
    body = SUBJ.sub("", body)
    body = body.lower().strip()
    m = LOTS.fullmatch(body)
    if m:
        return n + _sign(m.group(1)) * int(m.group(2)) * int(m.group(3))
    m = PER.fullmatch(body)
    if m:
        return n + _sign(m.group(1)) * int(m.group(2)) * int(m.group(3))
    m = SIMPLE.fullmatch(body)
    if m:
        return n + _sign(m.group(1)) * int(m.group(2))
    if META.search(clause) and not re.search(r"\d", clause):
        return n  # pure rounding note: no arithmetic of its own
    return None


def _pct(n, factor):
    """Percent steps are exact decimals: n * factor / 100."""
    return (n if isinstance(n, Decimal) else Decimal(n)) * factor / 100


def solve(instruction: str, task_dir: str):
    sents = _sentences(instruction)
    start = next((i for i, s in enumerate(sents) if _initial(s, task_dir) is not None), None)
    end = next((i for i, s in enumerate(sents) if s.endswith("?")), None)
    if start is None or end is None or end <= start:
        return None
    if any(re.search(r"\d", s) for s in sents[:start]):
        return None  # numbers before the recognised start: the chain is ambiguous
    n = _initial(sents[start], task_dir)
    if n is None:
        return None
    for sent in sents[start + 1:end]:
        for clause in _clauses(sent):
            n2 = _apply(n, clause)
            if n2 is None:
                return None  # unknown step: never guess
            n = n2
    if isinstance(n, Decimal):
        dm = ROUND_TO.search(instruction)
        if dm:
            return str(n.quantize(Decimal(1).scaleb(-int(dm.group(1))), rounding=ROUND_HALF_UP))
        return str(int(n)) if n == n.to_integral_value() else None
    return str(n)
