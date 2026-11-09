"""Deterministic 100-task synthetic benchmark (85 feasible + 15 infeasible).

Unlike tests/fixtures (copies of the real samples), this generates many variants
of the same formats with independently computed ground truth, so we can measure:

  * zero-token solver coverage (triage status == solved),
  * WRONG answer rate (solved but != expected) -- the metric that matters most,
  * infeasible detection rate for free.

The generator never calls the solvers: expected answers are computed by its own
straight-line arithmetic (math), plain csv/statistics (data) and dict walks
(search), so a solver bug cannot be masked by sharing the same code.

Usage:
  python tools/gen_synthetic.py /tmp/synth100
  python tools/bench_solvers.py /tmp/synth100
"""
import json
import math
import os
import random
import shutil

REGIONS = ["west", "north", "east", "south"]
PRODUCTS = ["widget", "gadget", "gizmo", "sprocket"]


def _write(out, name, typ, instruction, expected, reward=1):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    meta = {"task_id": name, "task_type": typ, "instruction": instruction, "reward": reward}
    if expected is None:
        meta["check"] = "infeasible"
    else:
        meta["expected"] = expected
    with open(os.path.join(d, "task.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return d


TAIL = (" Compute it with the bash tool (python3 -c). Work only inside the current directory. "
        "When you are done, write ONLY the final answer to the file answer.txt in the current directory "
        "(for example with bash: echo 42 > answer.txt).")


# ---------------- math ----------------

def _math_task(rnd, idx, out):
    start = rnd.randint(500, 4000)
    n = start
    parts = [f"A warehouse starts with {start} boxes."]
    splits = 0
    for _ in range(rnd.randint(3, 5)):
        kinds = ["mul", "ship", "receive", "lots", "buy", "selloff", "throw"]
        if splits < 2:
            kinds += ["split", "split2"]
        kind = rnd.choice(kinds)
        if kind in ("split", "split2"):
            k = rnd.choice([2, 3, 4])
            up = rnd.random() < 0.4
            n = -(-n // k) if up else n // k
            if kind == "split":
                parts.append(f"Then the stock is split equally into {k} warehouses and it keeps one share "
                             f"(rounded {'up' if up else 'down'}).")
            else:
                parts.append(f"Then the stock is divided equally into {k} parts and it keeps one portion "
                             f"(rounded {'up' if up else 'down'}).")
            splits += 1
        elif kind == "mul":
            w = rnd.choice([2, 3, 4])
            word = {2: "doubled", 3: "tripled", 4: "quadrupled"}[w]
            if rnd.random() < 0.5:
                parts.append(f"Then the stock is {word} by merging with a sister warehouse.")
            else:
                parts.append(f"Then the stock {word}.")
            n *= w
        elif kind == "ship":
            d = rnd.randint(2, 5)
            a = rnd.randint(1, max(1, n // (d * 3)))
            n -= a * d
            parts.append(f"Then it ships {a} boxes per day for {d} days.")
        elif kind == "receive":
            a = rnd.randint(5, 90)
            n += a
            parts.append(f"Then it receives {a} boxes.")
        elif kind == "lots":
            lots = rnd.randint(2, 4)
            a = rnd.randint(5, 80)
            n += lots * a
            parts.append(f"Then it receives {lots} deliveries of {a} boxes each.")
        elif kind == "buy":
            a = rnd.randint(5, 60)
            n += a
            parts.append(f"Then it buys {a} boxes.")
        elif kind == "selloff":
            a = rnd.randint(1, max(1, n // 4))
            n -= a
            parts.append(f"Then it sells off {a} boxes.")
        else:
            a = rnd.randint(1, max(1, n // 5))
            n -= a
            parts.append(f"Then it throws away {a} boxes.")
        if n < 10:
            n += 50
            parts.append("Then it receives 50 boxes.")
    parts.append("How many boxes does it have at the end?" + TAIL)
    return _write(out, f"synth_math_{idx:03d}", "math", " ".join(parts), str(n))


# ---------------- data ----------------

def _read_rows(d, name):
    import csv
    with open(os.path.join(d, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _data_task(rnd, idx, out):
    nrows = rnd.randint(24, 60)
    rows = []
    for i in range(nrows):
        if i < len(PRODUCTS):
            region, product = REGIONS[i], PRODUCTS[i]
        else:
            region, product = rnd.choice(REGIONS), rnd.choice(PRODUCTS)
        rows.append({"order_id": i + 1, "region": region, "product": product,
                     "month": rnd.randint(1, 12), "units": rnd.randint(1, 50), "price": rnd.randint(10, 100)})
    name = f"synth_data_{idx:03d}"
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    import csv
    with open(os.path.join(d, "sales.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pre = (f"The file sales.csv has columns order_id,region,product,month,units,price "
           f"({len(rows)} rows). ")
    kind = rnd.choice(["sum", "revenue", "count", "avg", "max", "min", "median",
                       "distinct", "percentage", "top_units", "top_count", "top_avg"])
    if kind == "sum":
        p, a, b = rnd.choice(PRODUCTS), rnd.randint(1, 6), rnd.randint(7, 12)
        q = f"How many units of product '{p}' were sold in months {a} through {b} inclusive?"
        exp = sum(r["units"] for r in rows if r["product"] == p and a <= r["month"] <= b)
        exp = str(exp)
    elif kind == "revenue":
        reg = rnd.choice(REGIONS)
        q = f"What is the total revenue (units * price) in region '{reg}'?"
        exp = str(sum(r["units"] * r["price"] for r in rows if r["region"] == reg))
    elif kind == "count":
        reg = rnd.choice(REGIONS)
        q = f"How many orders were placed in region '{reg}'?"
        exp = str(sum(1 for r in rows if r["region"] == reg))
    elif kind in ("avg", "median"):
        p = rnd.choice(PRODUCTS)
        vals = sorted(r["price"] for r in rows if r["product"] == p)
        if kind == "avg":
            q = f"What is the average price of product '{p}'? Round to 2 decimals."
            exp = f"{sum(vals) / len(vals):.2f}"
        else:
            q = f"What is the median price of product '{p}'? Round to 2 decimals."
            exp = f"{_median(vals):.2f}"
    elif kind in ("max", "min"):
        reg = rnd.choice(REGIONS)
        vals = [r["units"] for r in rows if r["region"] == reg]
        q = (f"What is the maximum units in region '{reg}'?" if kind == "max"
             else f"What is the minimum units in region '{reg}'?")
        exp = str(max(vals) if kind == "max" else min(vals))
    elif kind == "distinct":
        reg = rnd.choice(REGIONS)
        q = f"How many unique products were sold in region '{reg}'?"
        exp = str(len({r["product"] for r in rows if r["region"] == reg}))
    elif kind == "percentage":
        reg = rnd.choice(REGIONS)
        q = f"What percentage of orders were placed in region '{reg}'? Round to 1 decimal."
        exp = f"{100.0 * sum(1 for r in rows if r['region'] == reg) / len(rows):.1f}"
    elif kind == "top_units":
        q = "Which product had the highest total units?"
        sums = {p: sum(r["units"] for r in rows if r["product"] == p) for p in PRODUCTS}
        win = _winner(sums, max)
        if win is None:
            return _data_fallback(rows, d, name, pre)
        exp = win
    elif kind == "top_count":
        q = "Which region sold the most orders?"
        counts = {g: sum(1 for r in rows if r["region"] == g) for g in REGIONS}
        win = _winner(counts, max)
        if win is None:
            return _data_fallback(rows, d, name, pre)
        exp = win
    else:
        q = "Which region had the lowest average price?"
        means = {g: sum(r["price"] for r in rows if r["region"] == g) /
                    max(1, sum(1 for r in rows if r["region"] == g)) for g in REGIONS}
        win = _winner(means, min)
        if win is None:
            return _data_fallback(rows, d, name, pre)
        exp = win
    inst = pre + q + " Use bash with python3 to compute it. Work only inside the current directory. " \
                     "When you are done, write ONLY the final answer to the file answer.txt in the current " \
                     "directory (for example with bash: echo 42 > answer.txt)."
    return _write(out, name, "data_analysis", inst, exp)


def _median(vals):
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def _winner(d, fn):
    best = fn(d.values())
    wins = [k for k, v in d.items() if abs(v - best) < 1e-9]
    return wins[0] if len(wins) == 1 else None


def _data_fallback(rows, d, name, pre):
    reg = REGIONS[0]
    q = f"How many orders were placed in region '{reg}'?"
    exp = str(sum(1 for r in rows if r["region"] == reg))
    inst = pre + q + " Work only inside the current directory. When you are done, write ONLY the final " \
                     "answer to the file answer.txt in the current directory."
    meta_path = os.path.join(d, "task.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"task_id": name, "task_type": "data_analysis", "instruction": inst,
                   "reward": 1, "expected": exp}, f, indent=1)
    return d


# ---------------- search ----------------

SEARCH_DOCS = {
    "company_arktika.md": "# Arktika Labs\nHeadquarters: Medan\nFounded: 1982\nFounder: Boris Vale\nCEO: Alice Smith\n",
    "company_bluefin.md": "# Bluefin Systems\nHeadquarters: Medan\nFounded: 2018\nFounder: Chen Wei\nCEO: Alice Novak\n",
    "company_cobalt.md": "# Cobalt Works\nHeadquarters: Tbilisi\nFounded: 1968\nFounder: Boris Vale\nCEO: Kira Petrov\n",
    "company_dune.md": "# Dune Robotics\nHeadquarters: Hanoi\nFounded: 1989\nFounder: Chen Wei\nCEO: Olga Kowalski\nParent: Arktika Labs\n",
    "company_ember.md": "# Ember Bio\nHeadquarters: Osaka\nFounded: 1962\nFounder: Kira Petrov\nCEO: Alice Smith\nParent: Bluefin Systems\n",
    "person_alice_smith.md": "# Alice Smith\nBorn: 2002\nEmployer: Arktika Labs\nMentor: Olga Kowalski\n"
                             "Spouse: William King\nCountry: UK\n",
    "person_alice_novak.md": "# Alice Novak\nBorn: 1981\nEmployer: Bluefin Systems\nMentor: Kira Petrov\n"
                             "Spouse: Peter Novak\nCountry: Poland\n",
    "person_kira.md": "# Kira Petrov\nBorn: 1975\nEmployer: Cobalt Works\nMentor: Boris Vale\n"
                      "Spouse: Ivan Petrov\nCountry: Estonia\n",
    "person_olga.md": "# Olga Kowalski\nBorn: 1988\nEmployer: Dune Robotics\nMentor: Alice Smith\n"
                      "Spouse: Jan Kowalski\nCountry: Poland\n",
    "person_boris.md": "# Boris Vale\nBorn: 1950\nEmployer: Arktika Labs\nSpouse: Nina Vale\nCountry: Latvia\n",
    "person_chen.md": "# Chen Wei\nBorn: 1969\nEmployer: Bluefin Systems\nSpouse: Li Wei\nCountry: China\n",
    "book_notes.md": "# Notes on the Engine\nAuthor: Alice Smith\nDirector: Kira Petrov\n",
    "book_signals.md": "# Signals and Noise\nAuthor: Alice Novak\nDirector: Olga Kowalski\n",
    "book_rivers.md": "# Rivers of Glass\nAuthor: Kira Petrov\nDirector: Alice Smith\n",
}


def _search_docs():
    ents = {}
    for text in SEARCH_DOCS.values():
        lines = text.splitlines()
        title = lines[0][2:].strip()
        fields = {}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                fields[k.strip().lower()] = v.strip()
        ents[title] = fields
    return ents


def _search_task(rnd, idx, out):
    d = _write(out, f"synth_search_{idx:03d}", "search", "", "placeholder")
    kb = os.path.join(d, "kb")
    os.makedirs(kb)
    for fn, text in SEARCH_DOCS.items():
        with open(os.path.join(kb, fn), "w", encoding="utf-8") as f:
            f.write(text)
    ents = _search_docs()
    people = [e for e, f in ents.items() if "employer" in f and "born" in f]
    companies = [e for e, f in ents.items() if "headquarters" in f and "founded" in f]
    books = [e for e, f in ents.items() if "author" in f]
    person = rnd.choice(people)
    company = rnd.choice(companies)
    book = rnd.choice(books)
    kind = rnd.choice(["hq", "born", "mentor", "ceo", "parent", "author", "spouse", "country",
                       "founder", "director"])
    if kind == "hq":
        emp = ents[person]["employer"]
        q = f"In which city is the headquarters of {person}'s employer?"
        exp = ents[emp]["headquarters"]
    elif kind == "born":
        q = f"In which year was {person} born?"
        exp = ents[person]["born"]
    elif kind == "mentor":
        person = rnd.choice([p for p in people if "mentor" in ents[p]])
        q = f"Who is the mentor of {person}?"
        exp = ents[person]["mentor"]
    elif kind == "ceo":
        q = f"Who is the CEO of {company}?"
        exp = ents[company]["ceo"]
    elif kind == "parent":
        if "parent" not in ents[company]:
            comp2 = next(c for c in companies if "parent" in ents[c])
            company = comp2
        q = f"Who is the parent company of {company}?"
        exp = ents[company]["parent"]
    elif kind == "author":
        q = f"Who wrote {book}?"
        exp = ents[book]["author"]
    elif kind == "spouse":
        q = f"Who is {person} married to?"
        exp = ents[person]["spouse"]
    elif kind == "country":
        q = f"Which country is {person} a citizen of?"
        exp = ents[person]["country"]
    elif kind == "founder":
        q = f"Who is the founder of {company}?"
        exp = ents[company]["founder"]
    else:
        q = f"Who directed {book}?"
        exp = ents[book]["director"]
    inst = (f"The directory kb/ contains a small knowledge base of markdown files. {q} "
            "Use the file tools or bash (ls, grep, cat) to find it. Work only inside the current directory. "
            "When you are done, write ONLY the final answer to the file answer.txt in the current directory "
            "(for example with bash: echo 42 > answer.txt).")
    meta = {"task_id": f"synth_search_{idx:03d}", "task_type": "search", "instruction": inst,
            "reward": 1, "expected": exp}
    with open(os.path.join(d, "task.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return d


# ---------------- coding ----------------

BASE = {
    "reverse_words": "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
    "total": "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
    "maximum": "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
    "minimum": "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
    "average": "def average(xs):\n    return sum(xs) / len(xs)\n",
    "count_even": "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
}
TESTS = """import unittest
import mylib as m
class T(unittest.TestCase):
    def test_reverse_words(self): self.assertEqual(m.reverse_words('c a b'), 'b a c')
    def test_total(self): self.assertEqual(m.total([1, 2, 3]), 6)
    def test_maximum(self): self.assertEqual(m.maximum([3, 9, 2]), 9)
    def test_minimum(self): self.assertEqual(m.minimum([3, 9, 2]), 2)
    def test_average(self): self.assertAlmostEqual(m.average([1, 2, 4]), 7 / 3)
    def test_count_even(self): self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)
if __name__ == '__main__': unittest.main()
"""
BUGS = [("reverse_words", "reversed", "sorted"), ("total", "s += x", "s -= x"),
        ("maximum", "x > best", "x < best"), ("minimum", "x < best", "x > best"),
        ("average", "len(xs)", "(len(xs) + 1)"), ("count_even", "== 0", "== 1")]


def _coding_task(rnd, idx, out):
    d = _write(out, f"synth_code_{idx:03d}", "coding", "", "DONE")
    parts = dict(BASE)
    chosen = rnd.sample(BUGS, rnd.choice([1, 1, 2]))
    for fn, old, new in chosen:
        parts[fn] = parts[fn].replace(old, new, 1)
    with open(os.path.join(d, "mylib.py"), "w", encoding="utf-8") as f:
        f.write("\n\n".join(parts.values()))
    with open(os.path.join(d, "test_mylib.py"), "w", encoding="utf-8") as f:
        f.write(TESTS)
    inst = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
            "(do not edit the tests) until all tests pass. Work only inside the current directory. "
            "When you are done, write ONLY the word DONE to the file answer.txt in the current directory.")
    meta = {"task_id": f"synth_code_{idx:03d}", "task_type": "coding", "instruction": inst,
            "reward": 1, "expected": "DONE", "check": "unittest"}
    with open(os.path.join(d, "task.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return d


# ---------------- infeasible ----------------

def _infeasible(out, idx):
    kind = idx % 5
    name = f"synth_bad_{idx:03d}"
    if kind == 0:  # math: missing input file
        d = _write(out, name, "math", "A warehouse starts with 10 boxes. Use the starting stock from "
                                      "inventory.csv. How many boxes are left?" + TAIL, None)
    elif kind == 1:  # math: needs the internet
        d = _write(out, name, "math", "A warehouse starts with 10 boxes. Fetch the shipping rates from "
                                      "https://example.com/rates. How many boxes are left?" + TAIL, None)
    elif kind == 2:  # data: csv deleted
        d = _write(out, name, "data_analysis", "The file sales.csv has columns order_id,units (20 rows). "
                                               "How many units were sold in total? Work only inside the "
                                               "current directory.", None)
    elif kind == 3:  # search: entity missing
        d = _write(out, name, "search", "The directory kb/ contains a small knowledge base of markdown files. "
                                        "In which year was Zed Quark born? Work only inside the current "
                                        "directory.", None)
        kb = os.path.join(d, "kb")
        os.makedirs(kb)
        with open(os.path.join(kb, "a.md"), "w", encoding="utf-8") as f:
            f.write("# Alice Smith\nBorn: 2002\n")
    else:  # coding: module deleted
        d = _write(out, name, "coding", "Run `python3 test_mylib.py` in the current directory; some tests fail. "
                                        "Fix the bugs in mylib.py (do not edit the tests) until all tests pass. "
                                        "When you are done, write ONLY the word DONE to the file answer.txt.", None)
        with open(os.path.join(d, "test_mylib.py"), "w", encoding="utf-8") as f:
            f.write(TESTS)
    return d


def make(out_dir, seed=7):
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    rnd = random.Random(seed)
    for i in range(25):
        _math_task(rnd, i, out_dir)
    for i in range(25):
        _data_task(rnd, i, out_dir)
    for i in range(20):
        _search_task(rnd, i, out_dir)
    for i in range(15):
        _coding_task(rnd, i, out_dir)
    for i in range(15):
        _infeasible(out_dir, i)
    return out_dir


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth100"
    make(out)
    print(f"generated 100 tasks in {out}")
