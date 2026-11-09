"""Chain-of-operations word problems. Strict: any unknown sentence in the chain -> None."""
import math
import re

START = re.compile(r"starts? with (\d+)", re.I)
SPLIT = re.compile(r"(?:split|divid(?:e|es|ed)) equally into (\d+)[^.]*?keeps? (?:one|1) "
                   r"(?:share|part|portion|warehouse)?[^.]*?\(rounded (down|up)\)", re.I)
MULT = {"doubled": 2, "tripled": 3, "quadrupled": 4}
MULT_ANY = re.compile(r"\b(double|doubles|doubled|triple|triples|tripled|"
                      r"quadruple|quadruples|quadrupled)\b", re.I)
MULT_VAL = {v: k for word, k in (("double", 2), ("triple", 3), ("quadruple", 4))
            for v in (word, word + "s", word + "d")}
ADD = r"(?:receives?|gets?|adds?|gains?|takes? in|buys?|purchases?|restocks?)"
SUB = (r"(?:ships?(?: out)?|sells?(?: off)?|loses?|removes?|uses?|donates?|discards?|"
       r"throws? away|gives? away|sends? out)")
V = rf"({ADD}|{SUB})"
SUBJ = re.compile(r"^(?:then,?\s+|next,?\s+|after that,?\s+)?(?:it|the \w+(?: \w+)?)\s+", re.I)
PER = re.compile(rf"{V} (\d+) \w+ per \w+ for (\d+) \w+")
LOTS = re.compile(rf"{V} (\d+) (?:deliveries|delivery|shipments|shipment|batches|batch|orders|order|trucks|truck) of (\d+) \w+ each")
ONCE = re.compile(rf"{V} (\d+) \w+")


def _sign(verb: str) -> int:
    return 1 if re.fullmatch(ADD, verb) else -1


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def solve(instruction: str, task_dir: str):
    sents = _sentences(instruction)
    start = next((i for i, s in enumerate(sents) if START.search(s)), None)
    end = next((i for i, s in enumerate(sents) if s.endswith("?")), None)
    if start is None or end is None or end <= start:
        return None
    n = int(START.search(sents[start]).group(1))
    for s in sents[start + 1:end]:
        m = SPLIT.search(s)
        if m:
            k = int(m.group(1))
            if k <= 0:
                return None
            n = n // k if m.group(2).lower() == "down" else math.ceil(n / k)
            continue
        w = next((v for k, v in MULT.items() if k in s.lower()), None)
        if w and "merging" in s.lower():
            n *= w
            continue
        mm = MULT_ANY.search(s)
        if mm and not re.search(r"\b(?:every|each|per)\b", s, re.I):
            n *= MULT_VAL[mm.group(1).lower()]
            continue
        body = SUBJ.sub("", s.strip().rstrip(".").lower())
        m = PER.fullmatch(body)
        if m:
            n += _sign(m.group(1)) * int(m.group(2)) * int(m.group(3))
            continue
        m = LOTS.fullmatch(body)
        if m:
            n += _sign(m.group(1)) * int(m.group(2)) * int(m.group(3))
            continue
        m = ONCE.fullmatch(body)
        if m:
            n += _sign(m.group(1)) * int(m.group(2))
            continue
        return None  # unknown step: never guess
    return str(n)
