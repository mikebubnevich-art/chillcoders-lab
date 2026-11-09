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
    "employer": ["employer", "works for", "work for", "works at", "work at", "worked for", "employed by"],
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


def solve(instruction: str, task_dir: str):
    kb = os.path.join(task_dir, "kb")
    if not os.path.isdir(kb):
        return UNSOLVABLE if "kb/" in instruction else None
    ents, alltext = _load(kb)
    if not ents:
        return None
    q = instruction.split("Use the file tools")[0]
    low = q.lower()
    names = sorted(ents, key=len, reverse=True)
    ent = next((n for n in names if n.lower() in low), None)
    if ent is None:
        # a capitalised 'First Last' in the question that exists nowhere in the KB => infeasible
        cand = re.findall(r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", q)
        cand = [c for c in cand if c.split()[0] not in ("In", "The", "Who", "What", "Which")]
        if cand and all(c.lower() not in alltext for c in cand):
            return UNSOLVABLE
        return None
    keys = _field_keys(ents)
    mentioned = [k for k in keys if any(re.search(rf"\b{re.escape(s)}\b", low) for s in SYN.get(k, [k]))]
    # 'city' only matters together with headquarters; drop fields that are just the entity's own words
    link = [k for k in mentioned if any(v in ents for e in ents.values() for kk, v in e.items() if kk == k)]
    terminal = [k for k in mentioned if k not in link]
    if len(terminal) > 1 or not (link or terminal):
        return None
    results = set()
    for perm in itertools.permutations(link):
        cur, ok = ent, True
        for k in perm:
            nxt = ents.get(cur, {}).get(k)
            if nxt is None:
                ok = False
                break
            cur = nxt
        if not ok:
            continue
        if terminal:
            v = ents.get(cur, {}).get(terminal[0])
            if v is None:
                continue
            results.add(v)
        elif cur != ent:
            results.add(cur)
    if len(results) == 1:
        return results.pop()
    return None
