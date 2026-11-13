"""Deterministic 'hard but honest' benchmark: solvable tasks that break the easy grammar.

`gen_synthetic.py` builds the formats our solvers were written for, so it measures
regressions, not headroom. This generator builds the *neighbouring* formats that
look the same to a human but are outside the 0-token grammar:

  data  - prices as "$1,204.50", discount column with blanks, month names,
          "at least N" thresholds, an exact .x5 rounding tie, even-count median,
          an explicit tie-break rule, product labels with trailing spaces;
  math  - "half of the remaining", stock read from CSV, grams -> full bags,
          percent + tax, start value from JSON;
  search- reverse relations, 3-hop chains, entity titles with double spaces,
          ambiguous names (infeasible), missing entities (infeasible);
  coding- defects outside the single-mutation grammar (dropped .lower(),
          list(set()), wrong rotate direction, off-by-one loop, multi-defect);
  bad   - missing column, header-only file, contradictory row count, missing
          input file, network-only source, impossible tests, tie without rule.

Ground truth is computed here with plain csv/Decimal/arithmetic, never by the
solvers, so a solver bug cannot hide behind shared code.

Usage:
  python tools/gen_shadow.py /tmp/shadow
  python tools/bench_solvers.py /tmp/shadow
"""
import csv
import json
import os
import random
import re
import shutil
from decimal import Decimal, ROUND_HALF_UP

PRODUCTS = ["widget", "gadget", "gizmo", "sprocket"]
REGIONS = ["west", "north", "east", "south"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_FULL = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]

TAIL = (" Work only inside the current directory. When you are done, write ONLY the final "
        "answer to the file answer.txt in the current directory (for example with bash: "
        "echo 42 > answer.txt).")
BASH = " Use bash with python3 to compute it."


def _write(out, name, typ, instruction, expected, reward=1, check=None):
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    meta = {"task_id": name, "task_type": typ, "instruction": instruction, "reward": reward}
    if expected is None:
        meta["check"] = "infeasible"
    else:
        meta["expected"] = expected
        if check:
            meta["check"] = check
    with open(os.path.join(d, "task.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return d


def _csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _money(x) -> str:
    return str(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _parse_money(v: str) -> Decimal:
    return Decimal(str(v).replace("$", "").replace(",", "").strip() or "0")


_PRICE_STYLES = ["usd", "usd_word", "eur", "eu_space", "ch", "rub"]


def _eu(price: Decimal) -> str:
    """1204.50 -> '1.204,50' (dot groups, comma decimals)."""
    intpart, _, frac = f"{price:.2f}".partition(".")
    return f"{int(intpart):,}".replace(",", ".") + "," + frac


def _fmt_price(price: Decimal, style: str) -> str:
    if style == "usd":
        return "$" + f"{price:,.2f}"
    if style == "usd_word":
        return f"{price:.2f} USD"
    if style == "eur":
        return "\u20ac" + _eu(price)
    if style == "eu_space":
        return _eu(price).replace(".", " ")
    if style == "ch":
        return f"{price:,.2f}".replace(",", "'")
    if style == "rub":
        return "\u20bd" + _eu(price)
    raise ValueError(style)


def _parse_money_any(v):
    """Independent (from the solver) reader of the money formats the corpus
    emits: '$1,204.50', '\u20ac1.204,50', '1 204,50', "1'204.50", '1204.50 USD',
    '\u20bd1204,50', '20%'. Same separator rules: the rightmost of ','/'.' is
    the decimal separator; a lone separator with 3-digit groups is grouping."""
    s = str(v).strip()
    if not s:
        return Decimal(0)
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    for ch in "$\u20ac\u00a3\u00a5\u20bd\u20b9\u20ba()+\u2019'\u00a0":
        s = s.replace(ch, " ")
    s = re.sub(r"(?i)\b(usd|eur|gbp|chf|rub|rur|dollars?|euros?|pounds?|rubles?|francs?)\b", " ", s)
    s = re.sub(r"[\s%]+", "", s)
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        parts = s.split(",")
        if len(parts) > 1 and len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:]):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif "." in s:
        parts = s.split(".")
        if len(parts) > 1 and len(parts[0]) <= 3 and all(len(p) == 3 for p in parts[1:]):
            s = s.replace(".", "")
    return Decimal(s) * (-1 if neg else 1)


# ---------------- data_analysis ----------------

def data_discount(rnd, i, out):
    """Prices in one of several money formats; discount in percent, blank = none."""
    rows = []
    style = rnd.choice(_PRICE_STYLES)
    disc_pct = rnd.random() < 0.5  # discount cell as '20%' vs plain 20
    for k in range(rnd.randint(22, 34)):
        price = Decimal(rnd.randint(800, 2400)) + Decimal(rnd.choice(["0.50", "0.25", "0"]))
        disc = rnd.choice([0, 0, 5, 10, 15, 20])
        if disc == 0:
            dcell = ""
        elif disc_pct:
            dcell = f"{disc}%"
        else:
            dcell = disc
        rows.append({
            "order_id": k + 1,
            "region": rnd.choice(REGIONS),
            "product": rnd.choice(PRODUCTS),
            "units": rnd.randint(1, 9),
            "price": _fmt_price(price, style),
            "discount": dcell,
        })
    name = f"shadow_data_discount_{i:03d}"
    d = _write(out, name, "data_analysis", "", "0")
    _csv(os.path.join(d, "sales.csv"), rows)
    region = rnd.choice(REGIONS)
    total = Decimal(0)
    for r in rows:
        if r["region"] != region:
            continue
        units = Decimal(str(r["units"]))
        price = _parse_money_any(r["price"])
        dcell = str(r["discount"]).replace("%", "").strip()
        disc = Decimal(dcell) if dcell else Decimal(0)
        total += price * units * (Decimal(1) - disc / 100)
    q = (f"The file sales.csv has columns order_id,region,product,units,price,discount "
         f"({len(rows)} rows). Prices are written like \"{_fmt_price(Decimal('1204.50'), style)}\"; "
         f"discount is a percent and may be empty. What is the total revenue in region "
         f"'{region}', where revenue is units * price with the discount applied? "
         f"Round to 2 decimals.")
    return _rewrite(d, name, "data_analysis", q + BASH + TAIL, _money(total))


def data_months(rnd, i, out):
    rows = []
    for k in range(rnd.randint(24, 40)):
        rows.append({
            "order_id": k + 1,
            "region": rnd.choice(REGIONS),
            "product": rnd.choice(PRODUCTS),
            "month": rnd.choice(MONTHS),
            "units": rnd.randint(1, 40),
            "price": rnd.randint(10, 120),
        })
    name = f"shadow_data_months_{i:03d}"
    d = _write(out, name, "data_analysis", "", "0")
    _csv(os.path.join(d, "sales.csv"), rows)
    product, a, b = rnd.choice(PRODUCTS), 6, 7  # July, August
    lo, hi = MONTH_FULL[a], MONTH_FULL[b]
    total = sum(r["units"] for r in rows if r["product"] == product and r["month"] in (MONTHS[a], MONTHS[b]))
    q = (f"The file sales.csv has columns order_id,region,product,month,units,price "
         f"({len(rows)} rows). The month column holds three-letter month names. How many "
         f"units of product '{product}' were sold in months {lo} through {hi} inclusive?")
    return _rewrite(d, name, "data_analysis", q + BASH + TAIL, str(total))


def data_threshold(rnd, i, out):
    rows = []
    for k in range(rnd.randint(25, 45)):
        rows.append({
            "order_id": k + 1,
            "region": rnd.choice(REGIONS),
            "product": rnd.choice(PRODUCTS),
            "month": rnd.randint(1, 12),
            "units": rnd.randint(1, 60),
            "price": rnd.randint(5, 90),
        })
    name = f"shadow_data_threshold_{i:03d}"
    d = _write(out, name, "data_analysis", "", "0")
    _csv(os.path.join(d, "sales.csv"), rows)
    n = rnd.choice([20, 25, 30, 30])
    q = (f"The file sales.csv has columns order_id,region,product,month,units,price "
         f"({len(rows)} rows). How many orders have at least {n} units?")
    return _rewrite(d, name, "data_analysis", q + BASH + TAIL,
                    str(sum(1 for r in rows if r["units"] >= n)))


def data_percent_tie(rnd, i, out):
    """13 of 80 rows = 16.25% -> 16.3 with half-up, 16.2 with banker's rounding."""
    rows = []
    for k in range(80):
        rows.append({
            "order_id": k + 1,
            "region": "north" if k < 13 else rnd.choice(["west", "east", "south"]),
            "product": rnd.choice(PRODUCTS),
            "month": rnd.randint(1, 12),
            "units": rnd.randint(1, 20),
            "price": rnd.randint(5, 50),
        })
    name = f"shadow_data_percent_{i:03d}"
    d = _write(out, name, "data_analysis", "", "0")
    _csv(os.path.join(d, "sales.csv"), rows)
    q = (f"The file sales.csv has columns order_id,region,product,month,units,price "
         f"(80 rows). What percentage of orders were placed in region 'north'? "
         f"Round to 1 decimal.")
    pct = (Decimal(13) * Decimal(100) / Decimal(80)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return _rewrite(d, name, "data_analysis", q + BASH + TAIL, str(pct))


def data_median_even(rnd, i, out):
    rows = []
    prices = [11, 22, 33, 44, 55, 66][: rnd.choice([4, 6])]
    for k in range(30):
        rows.append({
            "order_id": k + 1,
            "region": rnd.choice(REGIONS),
            "product": "gizmo" if k < len(prices) else rnd.choice(["widget", "gadget", "sprocket"]),
            "month": rnd.randint(1, 12),
            "units": rnd.randint(1, 15),
            "price": prices[k] if k < len(prices) else rnd.randint(70, 99),
        })
    name = f"shadow_data_median_{i:03d}"
    d = _write(out, name, "data_analysis", "", "0")
    _csv(os.path.join(d, "sales.csv"), rows)
    vals = sorted(prices)
    n = len(vals)
    med = Decimal(vals[n // 2 - 1] + vals[n // 2]) / Decimal(2)
    q = (f"The file sales.csv has columns order_id,region,product,month,units,price "
         f"({len(rows)} rows). What is the median price of product 'gizmo'? "
         f"Round to 2 decimals.")
    return _rewrite(d, name, "data_analysis", q + BASH + TAIL, _money(med))


def data_tiebreak(rnd, i, out):
    rows = []
    # "gadget" and "widget" carry exactly the same total units: the rule decides.
    for k in range(28):
        product = "gadget" if k % 2 else "widget"
        rows.append({
            "order_id": k + 1,
            "region": rnd.choice(REGIONS),
            "product": product if k < 12 else rnd.choice(["gizmo", "sprocket"]),
            "month": rnd.randint(1, 12),
            "units": 10,
            "price": rnd.randint(5, 40),
        })
    name = f"shadow_data_tiebreak_{i:03d}"
    d = _write(out, name, "data_analysis", "", "0")
    _csv(os.path.join(d, "sales.csv"), rows)
    sums = {p: sum(r["units"] for r in rows if r["product"] == p) for p in PRODUCTS}
    best = max(sums.values())
    tied = sorted(p for p, v in sums.items() if v == best)
    q = (f"The file sales.csv has columns order_id,region,product,month,units,price "
         f"({len(rows)} rows). Which product had the highest total units? If several "
         f"products tie, answer with the alphabetically first of them.")
    return _rewrite(d, name, "data_analysis", q + BASH + TAIL, tied[0])


def data_whitespace(rnd, i, out):
    rows = []
    for k in range(30):
        product = "widget " if k % 3 == 0 else ("widget" if k % 3 == 1 else rnd.choice(["gadget", "gizmo"]))
        rows.append({
            "order_id": k + 1,
            "region": rnd.choice(REGIONS),
            "product": product,
            "month": rnd.randint(1, 12),
            "units": rnd.randint(1, 25),
            "price": rnd.randint(5, 60),
        })
    name = f"shadow_data_whitespace_{i:03d}"
    d = _write(out, name, "data_analysis", "", "0")
    _csv(os.path.join(d, "sales.csv"), rows)
    q = (f"The file sales.csv has columns order_id,region,product,month,units,price "
         f"({len(rows)} rows). Some product labels were entered with trailing spaces. "
         f"How many units of product 'widget' were sold in total?")
    return _rewrite(d, name, "data_analysis", q + BASH + TAIL,
                    str(sum(r["units"] for r in rows if r["product"].strip() == "widget")))


# ---------------- math ----------------

def math_half(rnd, i, out):
    start = rnd.choice([480, 640, 720, 960])
    n = start
    parts = [f"A warehouse starts with {start} boxes."]
    n //= 2
    parts.append("Then it ships out half of the remaining boxes.")
    for _ in range(rnd.randint(1, 2)):
        a = rnd.randint(2, 5)
        m = rnd.randint(10, 40)
        n += a * m
        parts.append(f"Then it receives {a} deliveries of {m} boxes each.")
    b = rnd.randint(5, 25)
    n -= b
    parts.append(f"Then it throws away {b} damaged boxes.")
    parts.append("How many boxes does it have at the end?")
    name = f"shadow_math_half_{i:03d}"
    d = _write(out, name, "math", "", "0")
    return _rewrite(d, name, "math", " ".join(parts) + BASH + TAIL, str(n))


def math_csv(rnd, i, out):
    start = rnd.randint(300, 900)
    other = rnd.randint(100, 900)
    rows = [{"item": "bolts", "stock": start}, {"item": "nuts", "stock": other}]
    name = f"shadow_math_csv_{i:03d}"
    d = _write(out, name, "math", "", "0")
    _csv(os.path.join(d, "inventory.csv"), rows)
    n = start
    a = rnd.randint(3, 6)
    m = rnd.randint(20, 60)
    n -= a * m
    b = rnd.randint(40, 120)
    n += b
    q = (f"A warehouse starts with the number of bolts listed in inventory.csv (column "
         f"item,stock). Then it ships out {a} orders of {m} bolts each. Then it receives "
         f"{b} more bolts. How many bolts are left?")
    return _rewrite(d, name, "math", q + BASH + TAIL, str(n))


def math_units(rnd, i, out):
    crates = rnd.randint(6, 14)
    per = rnd.choice([2500, 1800, 3200])
    bag = rnd.choice([400, 450, 600])
    grams = crates * per
    q = (f"A depot receives {crates} crates of flour, each holding {per} grams. The flour is "
         f"repacked into bags of {bag} grams. How many full bags can be filled?")
    name = f"shadow_math_units_{i:03d}"
    d = _write(out, name, "math", "", "0")
    return _rewrite(d, name, "math", q + BASH + TAIL, str(grams // bag))


def math_percent(rnd, i, out):
    price = rnd.randint(200, 900)
    disc, tax = rnd.choice([10, 15, 20]), rnd.choice([8, 9, 12])
    net = Decimal(price) * (Decimal(100 - disc)) / Decimal(100)
    gross = net * (Decimal(100 + tax)) / Decimal(100)
    q = (f"A jacket costs ${price}. It is discounted by {disc}%, then {tax}% sales tax is "
         f"added to the discounted price. What is the final price? Round to 2 decimals.")
    name = f"shadow_math_percent_{i:03d}"
    d = _write(out, name, "math", "", "0")
    return _rewrite(d, name, "math", q + BASH + TAIL, _money(gross))


def math_json(rnd, i, out):
    start = rnd.randint(400, 1200)
    name = f"shadow_math_json_{i:03d}"
    d = _write(out, name, "math", "", "0")
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"start_stock": start, "warehouse": "north"}, f)
    n = start
    a = rnd.randint(2, 4)
    m = rnd.randint(25, 70)
    n -= a * m
    b = rnd.randint(30, 90)
    n += b
    q = (f"A warehouse starts with the stock given in config.json (key start_stock). It ships "
         f"out {a} orders of {m} boxes each, then receives {b} boxes. How many boxes are left?")
    return _rewrite(d, name, "math", q + BASH + TAIL, str(n))


# ---------------- search ----------------

PEOPLE = [
    ("Alice Moreau", {"born": "1974, Lyon", "employer": "Northwind Traders", "mentor": "Ivan Petrov",
                      "spouse": "Bruno Sala", "country": "France"}),
    ("Ivan Petrov", {"born": "1968, Omsk", "employer": "Volga Analytics", "country": "Russia"}),
    ("Marta  Ruiz", {"born": "1981, Cadiz", "employer": "Solstice Foods", "spouse": "Alice Moreau",
                     "country": "Spain"}),
    ("Kenji Sato", {"born": "1979, Kobe", "employer": "Kaido Robotics", "country": "Japan"}),
    ("Lena Fischer", {"born": "1985, Bremen", "employer": "Northwind Traders", "country": "Germany"}),
    ("Omar Haddad", {"born": "1970, Tunis", "employer": "Solstice Foods", "country": "Tunisia"}),
    ("Priya Nair", {"born": "1990, Kochi", "employer": "Kaido Robotics", "mentor": "Kenji Sato",
                    "country": "India"}),
    ("Tomas Vega", {"born": "1983, Lima", "employer": "Volga Analytics", "country": "Peru"}),
]
COMPANIES = [
    ("Northwind Traders", {"headquarters": "Rotterdam", "founded": "1952", "ceo": "Lena Fischer",
                           "parent": "Meridian Holdings"}),
    ("Solstice Foods", {"headquarters": "Valencia", "founded": "1968", "ceo": "Omar Haddad",
                        "parent": "Meridian Holdings"}),
    ("Kaido Robotics", {"headquarters": "Kobe", "founded": "1994", "ceo": "Kenji Sato",
                        "parent": "Takara Group"}),
    ("Volga Analytics", {"headquarters": "Kazan", "founded": "2004", "ceo": "Tomas Vega",
                         "parent": "Takara Group"}),
    ("Meridian Holdings", {"headquarters": "Zurich", "founded": "1931", "ceo": "Alice Moreau"}),
    ("Takara Group", {"headquarters": "Osaka", "founded": "1946", "ceo": "Kenji Sato"}),
]
BOOKS = [
    ("The Quiet Ledger", {"author": "Marta  Ruiz", "director": "Lena Fischer"}),
    ("Northern Lights", {"author": "Alice Moreau", "director": "Kenji Sato"}),
    ("Glass Harbour", {"author": "Priya Nair", "director": "Omar Haddad"}),
]


def _kb(d):
    kb = os.path.join(d, "kb")
    os.makedirs(kb, exist_ok=True)
    for title, fields in PEOPLE + COMPANIES + BOOKS:
        lines = [f"# {title}"] + [f"{k.capitalize()}: {v}" for k, v in fields.items()]
        fn = title.replace(" ", "_").replace("___", "__") + ".md"
        with open(os.path.join(kb, fn), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return {t: f for t, f in PEOPLE + COMPANIES + BOOKS}


def _search_task(out, name, q, expected, expected_marker=None):
    d = _write(out, name, "search", "", expected)
    _kb(d)
    inst = ("The directory kb/ contains a small knowledge base of markdown files. " + q +
            " Use the file tools or bash (ls, grep, cat) to find it. Work only inside the "
            "current directory." + TAIL)
    return _rewrite(d, name, "search", inst, expected)


def search_reverse(rnd, i, out):
    q = ("Which company employs the person who was born in Lyon? Answer with the company name.")
    return _search_task(out, f"shadow_search_reverse_{i:03d}", q, "Northwind Traders")


def search_chain(rnd, i, out):
    q = "In which city is the headquarters of Alice Moreau's employer?"
    return _search_task(out, f"shadow_search_chain_{i:03d}", q, "Rotterdam")


def search_space(rnd, i, out):
    q = "Who is the spouse of Marta Ruiz?"
    return _search_task(out, f"shadow_search_space_{i:03d}", q, "Alice Moreau")


def search_multi3(rnd, i, out):
    q = ("Who is the CEO of the parent company of Priya Nair's employer? "
         "Answer with the person's name.")
    return _search_task(out, f"shadow_search_multi3_{i:03d}", q, "Kenji Sato")


def search_ambiguous(rnd, i, out):
    d = _write(out, f"shadow_search_amb_{i:03d}", "search", "", None)
    ents = _kb(d)
    with open(os.path.join(d, "kb", "John_Smith.md"), "w", encoding="utf-8") as f:
        f.write("# John Smith\nEmployer: Solstice Foods\nBorn: 1971, Oslo\n")
    with open(os.path.join(d, "kb", "John__Smith.md"), "w", encoding="utf-8") as f:
        f.write("# John  Smith\nEmployer: Kaido Robotics\nBorn: 1983, Bergen\n")
    inst = ("The directory kb/ contains a small knowledge base of markdown files. Who is the "
            "employer of John Smith? Use the file tools or bash (ls, grep, cat) to find it. "
            "Work only inside the current directory." + TAIL)
    return _rewrite(d, f"shadow_search_amb_{i:03d}", "search", inst, None)


def search_missing(rnd, i, out):
    q = "Who is the mentor of Zara Quinn?"
    return _search_task(out, f"shadow_search_missing_{i:03d}", q, None)


# ---------------- coding ----------------

def _code_task(out, name, src, test, marker="DONE"):
    d = _write(out, name, "coding", "", marker, check="unittest")
    with open(os.path.join(d, "mylib.py"), "w", encoding="utf-8") as f:
        f.write(src)
    with open(os.path.join(d, "test_mylib.py"), "w", encoding="utf-8") as f:
        f.write(test)
    inst = (f"Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs "
            f"in mylib.py (do not edit the tests) until all tests pass. Work only inside the "
            f"current directory. When you are done, write ONLY the word {marker} to the file "
            f"answer.txt in the current directory (for example with bash: echo {marker} > answer.txt).")
    return _rewrite(d, name, "coding", inst, marker)


def code_palindrome(rnd, i, out):
    src = ("def is_palindrome(text):\n"
           "    cleaned = ''.join(ch for ch in text if ch.isalnum())\n"
           "    return cleaned == cleaned[::-1]\n")
    test = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
            "    def test_mixed_case(self):\n"
            "        self.assertTrue(mylib.is_palindrome('A man, a plan, a canal: Panama!'))\n"
            "    def test_negative(self):\n"
            "        self.assertFalse(mylib.is_palindrome('Hello'))\n\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n")
    return _code_task(out, f"shadow_code_pal_{i:03d}", src, test)


def code_dedupe(rnd, i, out):
    src = ("def unique_in_order(items):\n"
           "    return list(set(items))\n")
    test = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
            "    def test_order(self):\n"
            "        self.assertEqual(mylib.unique_in_order([3, 1, 3, 2, 1]), [3, 1, 2])\n"
            "    def test_empty(self):\n"
            "        self.assertEqual(mylib.unique_in_order([]), [])\n\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n")
    return _code_task(out, f"shadow_code_dedupe_{i:03d}", src, test)


def code_rotate(rnd, i, out):
    src = ("def rotate(items, steps):\n"
           "    if not items:\n"
           "        return items\n"
           "    steps = steps % len(items)\n"
           "    return items[steps:] + items[:steps]\n")
    test = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
            "    def test_right(self):\n"
            "        self.assertEqual(mylib.rotate([1, 2, 3, 4], 1), [4, 1, 2, 3])\n"
            "    def test_negative(self):\n"
            "        self.assertEqual(mylib.rotate([1, 2, 3], -1), [2, 3, 1])\n\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n")
    return _code_task(out, f"shadow_code_rot_{i:03d}", src, test, marker="PASS")


def code_rle(rnd, i, out):
    # Defect outside the single-mutation grammar: the run counter is never reset,
    # so every run after the first is merged. Needs a new statement, not an operator flip.
    src = ("def encode(text):\n"
           "    if not text:\n"
           "        return ''\n"
           "    parts, count = [], 1\n"
           "    for i in range(1, len(text)):\n"
           "        if text[i] == text[i - 1]:\n"
           "            count += 1\n"
           "        else:\n"
           "            parts.append(text[i - 1] + str(count))\n"
           "    if count > 1:\n"
           "        parts.append(text[-1] + str(count))\n"
           "    return ''.join(parts)\n")
    test = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
            "    def test_basic(self):\n"
            "        self.assertEqual(mylib.encode('aaabbc'), 'a3b2c1')\n"
            "    def test_single(self):\n"
            "        self.assertEqual(mylib.encode('abc'), 'a1b1c1')\n\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n")
    return _code_task(out, f"shadow_code_rle_{i:03d}", src, test)


def code_clamp(rnd, i, out):
    # Defect outside the mutation grammar: the bounds are applied in the wrong
    # order (min(low, ...) wins over the value), so only "too low" is clamped.
    src = ("def clamp(value, low, high):\n"
           "    return min(low, max(value, high))\n")
    test = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
            "    def test_inside(self):\n"
            "        self.assertEqual(mylib.clamp(5, 0, 10), 5)\n"
            "    def test_low(self):\n"
            "        self.assertEqual(mylib.clamp(-3, 0, 10), 0)\n"
            "    def test_high(self):\n"
            "        self.assertEqual(mylib.clamp(42, 0, 10), 10)\n\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n")
    return _code_task(out, f"shadow_code_clamp_{i:03d}", src, test)


def code_multi(rnd, i, out):
    src = ("def is_palindrome(text):\n"
           "    cleaned = ''.join(ch for ch in text if ch.isalnum())\n"
           "    return cleaned == cleaned[::-1]\n\n\n"
           "def unique_in_order(items):\n"
           "    return list(set(items))\n")
    test = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
            "    def test_palindrome(self):\n"
            "        self.assertTrue(mylib.is_palindrome('A man, a plan, a canal: Panama!'))\n"
            "    def test_dedupe(self):\n"
            "        self.assertEqual(mylib.unique_in_order([3, 1, 3, 2, 1]), [3, 1, 2])\n\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n")
    return _code_task(out, f"shadow_code_multi_{i:03d}", src, test)


# ---------------- infeasible ----------------

def bad_missing_column(rnd, i, out):
    name = f"shadow_bad_column_{i:03d}"
    d = _write(out, name, "data_analysis", "", None)
    rows = [{"order_id": k, "region": rnd.choice(REGIONS), "product": rnd.choice(PRODUCTS),
             "month": 1, "units": 3, "price": 9} for k in range(1, 20)]
    _csv(os.path.join(d, "sales.csv"), rows)
    q = (f"The file sales.csv has columns order_id,region,product,month,units,price "
         f"({len(rows)} rows). What is the sum of the revenue column?")
    return _rewrite(d, name, "data_analysis", q + BASH + TAIL, None)


def bad_empty(rnd, i, out):
    name = f"shadow_bad_empty_{i:03d}"
    d = _write(out, name, "data_analysis", "", None)
    with open(os.path.join(d, "sales.csv"), "w", encoding="utf-8") as f:
        f.write("order_id,region,product,month,units,price\n")
    q = ("The file sales.csv has columns order_id,region,product,month,units,price. "
         "What is the average price in region 'north'? Round to 2 decimals.")
    return _rewrite(d, name, "data_analysis", q + BASH + TAIL, None)


def bad_rowcount(rnd, i, out):
    name = f"shadow_bad_rowcount_{i:03d}"
    d = _write(out, name, "data_analysis", "", None)
    rows = [{"order_id": k, "region": rnd.choice(REGIONS), "product": rnd.choice(PRODUCTS),
             "month": rnd.randint(1, 12), "units": rnd.randint(1, 9), "price": 10} for k in range(1, 26)]
    _csv(os.path.join(d, "sales.csv"), rows)
    q = ("The file sales.csv has columns order_id,region,product,month,units,price (40 rows). "
         "How many orders were placed in region 'north'?")
    return _rewrite(d, name, "data_analysis", q + BASH + TAIL, None)


def bad_missing_file(rnd, i, out):
    name = f"shadow_bad_file_{i:03d}"
    d = _write(out, name, "math", "", None)
    q = ("A warehouse starts with the stock listed in inventory.csv. Then it ships out 3 "
         "orders of 40 bolts each. How many bolts are left?")
    return _rewrite(d, name, "math", q + BASH + TAIL, None)


def bad_network(rnd, i, out):
    name = f"shadow_bad_net_{i:03d}"
    d = _write(out, name, "math", "", None)
    q = ("A courier starts with 500 parcels. Fetch the shipping rates from "
         "https://example.com/rates and subtract them from the parcel count. "
         "How many parcels remain?")
    return _rewrite(d, name, "math", q + BASH + TAIL, None)


def code_missing_function(rnd, i, out):
    """Tests call `mylib.solve_all` — a name the module never had. The contract
    forbids editing the tests but not adding code to the module, so the task is
    feasible: implement the function to satisfy the tests (LLM fallback)."""
    name = f"shadow_code_missing_{i:03d}"
    src = ("def add(a, b):\n    return a + b\n")
    test = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
            "    def test_add(self):\n        self.assertEqual(mylib.add(1, 2), 3)\n"
            "    def test_missing(self):\n        self.assertEqual(mylib.solve_all([1, 2]), 3)\n\n\n"
            "if __name__ == '__main__':\n    unittest.main()\n")
    return _code_task(out, name, src, test)


def bad_tie_no_rule(rnd, i, out):
    name = f"shadow_bad_tie_{i:03d}"
    d = _write(out, name, "data_analysis", "", None)
    rows = []
    for k in range(26):
        product = "widget" if k % 2 else "gadget"
        rows.append({"order_id": k, "region": rnd.choice(REGIONS), "product": product,
                     "month": rnd.randint(1, 12), "units": 10, "price": 5})
    _csv(os.path.join(d, "sales.csv"), rows)
    q = ("The file sales.csv has columns order_id,region,product,month,units,price "
         "(26 rows). Which product had the highest total units?")
    return _rewrite(d, name, "data_analysis", q + BASH + TAIL, None)


def bad_logic(rnd, i, out):
    name = f"shadow_bad_logic_{i:03d}"
    d = _write(out, name, "logic", "", None)
    q = ("A bag contains exactly 10 marbles. There are 4 red marbles and 7 blue marbles, and "
         "no other marbles. How many green marbles are there? If the task cannot be solved, "
         "write UNSOLVABLE to answer.txt.")
    return _rewrite(d, name, "logic", q + TAIL, None)


# ---------------- plumbing ----------------

def _rewrite(d, name, typ, instruction, expected):
    meta = {"task_id": name, "task_type": typ, "instruction": instruction, "reward": 1}
    if expected is None:
        meta["check"] = "infeasible"
    else:
        meta["expected"] = expected
        if typ == "coding":
            meta["check"] = "unittest"
    with open(os.path.join(d, "task.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return d


DATA = [data_discount, data_months, data_threshold, data_percent_tie,
        data_median_even, data_tiebreak, data_whitespace]
MATH = [math_half, math_csv, math_units, math_percent, math_json]
SEARCH = [search_reverse, search_chain, search_space, search_multi3]
CODE = [code_palindrome, code_dedupe, code_rotate, code_rle, code_clamp, code_multi,
        code_missing_function]
BAD = [bad_missing_column, bad_empty, bad_rowcount, bad_missing_file, bad_network,
       bad_tie_no_rule, bad_logic,
       search_ambiguous, search_missing]


def make(out_dir, seed=11, variants=4):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rnd = random.Random(seed)
    made = []
    for group in (DATA, MATH, SEARCH, CODE, BAD):
        for fn in group:
            for i in range(variants):
                made.append(fn(rnd, i, out_dir))
    return made


if __name__ == "__main__":
    import sys

    dest = sys.argv[1] if len(sys.argv) > 1 else "/tmp/shadow"
    dirs = make(dest)
    print(f"generated {len(dirs)} tasks in {dest}")
