"""Chain-of-operations word problems. Strict: any unknown sentence in the chain -> None."""
import math
import re

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
# "remove one third of the water (round the amount removed down ...)"
FRACTION = re.compile(r"\b(?:one|1)[\s-]+(third|half|quarter)\s+of\b", re.I)
FRACTION_VERB = re.compile(r"\b(?:remove|removes|removed|take|takes|took|sell|sells|sold|use|uses|used|"
                           r"lose|loses|lost|spend|spends|spent|discard|discards|discarded)\b", re.I)
BUNDLE = re.compile(r"\bpacked\s+into\s+(?:bundles?|packs?|boxes?)\s+of\s+(\d+)\b", re.I)
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


def _initial(sent):
    m = HAS_LOTS.search(sent)
    if m:
        return int(m.group(1)) * int(m.group(2))
    m = START.search(sent)
    if m:
        return int(m.group(1))
    m = HAS.search(sent)
    if m:
        return int(m.group(1))
    return None


def _apply(n, clause):
    """One chain step; returns the new value or None when the step is unknown."""
    low = clause.lower()
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


def solve(instruction: str, task_dir: str):
    sents = _sentences(instruction)
    start = next((i for i, s in enumerate(sents) if _initial(s) is not None), None)
    end = next((i for i, s in enumerate(sents) if s.endswith("?")), None)
    if start is None or end is None or end <= start:
        return None
    if any(re.search(r"\d", s) for s in sents[:start]):
        return None  # numbers before the recognised start: the chain is ambiguous
    n = _initial(sents[start])
    if n is None:
        return None
    for sent in sents[start + 1:end]:
        for clause in _clauses(sent):
            n2 = _apply(n, clause)
            if n2 is None:
                return None  # unknown step: never guess
            n = n2
    return str(n)
