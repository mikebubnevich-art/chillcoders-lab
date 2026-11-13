"""Knowledge-base lookups over kb/*.md ('# Name' title + 'Key: value' lines).

Question -> (entity, set of mentioned fields). Fields are split into LINK fields
(value is another entity's title, e.g. Employer) and TERMINAL fields (Born,
Headquarters...). We try every ordering of the mentioned link fields followed by
the one terminal field and accept the result only if it is unique.
"""
import itertools
import os
import re

from . import UNSOLVABLE

SYN = {
    "headquarters": ["headquarters", "headquartered", "head office", "hq", "based in", "based", "city"],
    "location": ["location", "located in", "situated in", "based in"],
    "born": ["born", "birth", "birthplace", "date of birth"],
    "founded": ["founded", "establishment", "established", "foundation year"],
    "founder": ["founder", "founders", "co-founder", "founded by", "who founded", "started by"],
    "employer": [
        "employer", "works for", "work for", "works at", "work at",
        "worked for", "employed by", "employs", "employ", "company where"
    ],
    "mentor": ["mentor", "mentored"],
    "ceo": ["ceo", "chief executive", "chief executive officer", "led by", "run by"],
    "author": ["author", "written by", "wrote", "writer"],
    "spouse": ["spouse", "husband", "wife", "married to"],
    "parent": ["parent", "parent company"],
    "director": ["director", "directed", "directed by"],
    "capital": ["capital", "capital city"],
    "country": ["country", "nationality", "citizen of"],
}


PROSE_FIELD = {"born": "born", "founded": "founded", "established": "founded",
               "launched": "founded", "created": "founded", "formed": "founded",
               "started": "founded"}
ENTITY_FACT = re.compile(
    r"(?P<ent>[A-Z][A-Za-z.'&-]*(?:\s+[A-Z][A-Za-z.'&-]*){0,3})\s+"
    r"(?:was|is|were|are)\s+(?P<field>born|founded|established|launched|created|formed|started)\s+in\s+"
    r"(?P<val>\d{4})")
TITLE_FACT = re.compile(
    r"\b(?:founded|established|launched|created|formed|began|started)\s+in\s+(\d{4})", re.I)


def _load(kb):
    ents, text = {}, []
    for root, _, files in os.walk(kb):
        for fn in sorted(files):
            if not fn.lower().endswith(".md"):
                continue
            with open(os.path.join(root, fn), encoding="utf-8", errors="replace") as f:
                t = f.read()
            text.append(t.lower())
            lines = t.splitlines()
            title = next((l[2:].strip() for l in lines if l.startswith("# ")), None)
            if not title:
                continue
            fields = {}
            for l in lines:
                m = re.match(r"\s*[-*]?\s*([A-Za-z ]+?)\s*:\s*(.+?)\s*$", l)
                if m:
                    fields[m.group(1).strip().lower()] = m.group(2).strip()
            ents.setdefault(title, {}).update(fields)
            # Prose facts: "Jonas Reed was born in 1987." -> entity card.
            for line in lines:
                for sentence in re.split(r"(?<=[.?!])\s+", line):
                    for m in ENTITY_FACT.finditer(sentence):
                        name = m.group("ent").strip()
                        if name == title or name.split()[0] in ("The", "This", "It"):
                            continue
                        ents.setdefault(name, {}).setdefault(
                            PROSE_FIELD[m.group("field").lower()], m.group("val"))
            # Facts stated about the file's own title: "# Cedarline Systems" +
            # "Founded in 2004." -> founded=2004.
            for m in TITLE_FACT.finditer(t):
                ents[title].setdefault("founded", m.group(1))
    return ents, "\n".join(text)


def _field_keys(ents):
    return {k for f in ents.values() for k in f}


def _norm(s: str) -> str:
    """Collapse whitespace and case: 'Marta  Ruiz' and 'Marta Ruiz' are one name."""
    return re.sub(r"\s+", " ", s).strip().lower()


def _canonical_field(key: str):
    group = {
        "headquarters": "location",
        "location": "location",
        "capital": "location",
        "country": "country",
        "nationality": "country",
        "citizen of": "country",
        "born": "born",
        "birthplace": "born",
        "founded": "founded",
        "established": "founded",
        "founder": "founder",
        "employer": "employer",
        "mentor": "mentor",
        "ceo": "ceo",
        "author": "author",
        "spouse": "spouse",
        "parent": "parent",
        "director": "director",
    }
    return group.get(key, key)


def _reverse_clause(q: str):
    """'...the person who was born in Lyon?' -> ('born', 'Lyon').

    Works on the original case: the value must be a proper noun, so generic
    phrases ('the founder of X') can never be mistaken for a filter value.
    """
    part = re.split(r"\bwho\b", q, maxsplit=1)
    if len(part) != 2:
        return None
    part = part[1].split("?")[0].split(". ")[0]
    for key, syns in SYN.items():
        for syn in syns:
            m = re.search(rf"(?<![A-Za-z0-9]){re.escape(syn)}(?![A-Za-z0-9])", part, re.I)
            if not m:
                continue
            tail = part[m.end():].strip(" ,;")
            v = re.match(r"(?:in|at|to|by|of)\s+([A-Z][\w'&.-]*(?:\s+[A-Z][\w'&.-]*)*)", tail)
            if v and not v.group(1).split()[0] in ("The", "This", "It"):
                return key, v.group(1)
    return None


def _candidates(key, value, ents):
    """Entities whose canonical field `key` mentions `value` (word match)."""
    want = _norm(value)
    pat = rf"(?<![a-z0-9]){re.escape(want)}(?![a-z0-9])"
    canon = _canonical_field(key)
    out = []
    for name, f in ents.items():
        for kk, vv in f.items():
            if _canonical_field(kk) == canon and re.search(pat, _norm(str(vv))):
                out.append(name)
                break
    return out


def _traverse(start, link, terminal, ents):
    cur, ok = start, True
    for k in link:
        nxt = ents.get(cur, {}).get(k)
        if nxt is None:
            return None
        cur = nxt
    if terminal:
        v = ents.get(cur, {}).get(terminal[0])
        return None if v is None else v
    return None if cur == start else cur


def solve(instruction: str, task_dir: str):
    kb = os.path.join(task_dir, "kb")
    if not os.path.isdir(kb):
        return UNSOLVABLE if "kb/" in instruction else None
    ents, alltext = _load(kb)
    if not ents:
        return None
    q = instruction.split("Use the file tools")[0]
    low = _norm(q)
    by_norm = {}
    for name in ents:
        by_norm.setdefault(_norm(name), []).append(name)
    # longest normalized title mentioned in the question (titles may differ from
    # the question only by spacing: 'Marta  Ruiz' vs 'Marta Ruiz')
    names = sorted(by_norm, key=len, reverse=True)
    ent_norm = next((n for n in names if n in low), None)
    reverse_cands = None
    reverse_key = None
    if ent_norm is None:
        # Reverse lookup: no entity is named, the start is described by a field
        # value ("Which company employs the person who was born in Lyon?").
        filt = _reverse_clause(q)
        if filt:
            reverse_key, reverse_val = filt
            reverse_cands = _candidates(reverse_key, reverse_val, ents)
            if not reverse_cands:
                return UNSOLVABLE  # no such entity in the KB: provably infeasible
            ent_norm = _norm(reverse_cands[0])
        else:
            # a capitalised 'First Last' in the question that exists nowhere in the KB => infeasible
            cand = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", q)
            cand = [c for c in cand if c.split()[0] not in ("In", "The", "Who", "What", "Which")]
            if cand and all(_norm(c) not in {_norm(t) for t in ents} for c in cand):
                return UNSOLVABLE
            return None
    if len(by_norm[ent_norm]) > 1:
        # Two cards share one name with different facts: no unique answer.
        return UNSOLVABLE
    ent = by_norm[ent_norm][0]
    keys = _field_keys(ents)
    matched = {}
    for key in keys:
        score = 0
        for syn in SYN.get(key, [key]):
            pattern = rf"(?<![A-Za-z0-9]){re.escape(syn)}(?![A-Za-z0-9])"
            if re.search(pattern, low):
                score += 2
        if score:
            canonical = _canonical_field(key)
            prev = matched.get(canonical)
            current = (score, {"headquarters": 3, "location": 2, "born": 3, "founded": 3}.get(key, 1), key)
            if prev is None or current > prev:
                matched[canonical] = current
    mentioned = [key for _, _, key in sorted(matched.values(), key=lambda x: (-x[0], -x[1], x[2]))]
    if reverse_key:
        # the filter field describes the start entity, not the answer
        # ('... the person who was born in Lyon?' -> born is not the target)
        mentioned = [k for k in mentioned if _canonical_field(k) != _canonical_field(reverse_key)]
    # 'city' only matters together with headquarters; drop fields that are just the entity's own words
    link = [k for k in mentioned if any(v in ents for e in ents.values() for kk, v in e.items() if kk == k)]
    terminal = [k for k in mentioned if k not in link]
    if len(terminal) > 1 or not (link or terminal):
        return None
    starts = reverse_cands if reverse_cands is not None else [ent]
    results = set()
    for start in starts:
        for perm in itertools.permutations(link):
            v = _traverse(start, perm, terminal, ents)
            if v is not None:
                results.add(v)
    if len(results) == 1:
        return results.pop()
    return None
