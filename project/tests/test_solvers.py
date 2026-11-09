import csv, os, shutil, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from solvers import UNSOLVABLE, data_solver, math_solver, search_solver

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/tasks")
PRE = "The file sales.csv has columns order_id,region,product,month,units,price (20 rows). "
POST = " Work only inside the current directory."


class Math(unittest.TestCase):
    def q(self, start, steps):
        return f"A warehouse starts with {start} boxes. " + " ".join(steps) + " How many boxes does it have at the end?"

    def test_sample(self):
        s = ["Then the stock is split equally into 2 warehouses and it keeps one share (rounded down).",
             "Then the stock is doubled by merging with a sister warehouse.",
             "Then the stock is split equally into 3 warehouses and it keeps one share (rounded down).",
             "Then the stock is doubled by merging with a sister warehouse.",
             "Then the stock is split equally into 3 warehouses and it keeps one share (rounded down)."]
        self.assertEqual(math_solver.solve(self.q(1752, s), ""), "389")

    def test_variants(self):
        s = ["Then the stock is split equally into 4 warehouses and it keeps one share (rounded up).",
             "Then the stock is tripled by merging with a sister warehouse."]
        self.assertEqual(math_solver.solve(self.q(10, s), ""), str(-(-10 // 4) * 3))

    def test_unknown_step_falls_back(self):
        self.assertIsNone(math_solver.solve(self.q(10, ["Then 5 boxes are lost."]), ""))

    def test_buy_and_dispose_verbs(self):
        s = ["Then it buys 40 boxes.", "Then it purchases 15 boxes.",
             "Then it restocks 25 boxes.", "Then it sells off 30 boxes.",
             "Then it throws away 10 boxes."]
        self.assertEqual(math_solver.solve(self.q(100, s), ""), str(100 + 40 + 15 + 25 - 30 - 10))

    def test_split_wording_variants(self):
        for step, expected in [
            ("Then the stock is divided equally into 3 warehouses and it keeps one portion (rounded down).", 10 // 3),
            ("Then the stock is split equally into 4 warehouses and it keeps 1 share (rounded up).", -(-10 // 4)),
            ("Then the stock is divided equally into 3 parts and it keeps one part (rounded down).", 10 // 3),
        ]:
            with self.subTest(step=step):
                self.assertEqual(math_solver.solve(self.q(10, [step]), ""), str(expected))

    def test_multiplier_without_merging(self):
        for word, k in [("doubles", 2), ("triples", 3), ("quadruples", 4)]:
            with self.subTest(word=word):
                self.assertEqual(math_solver.solve(self.q(7, [f"Then the stock {word}."]), ""), str(7 * k))

    def test_per_period_multiplier_is_not_a_chain_step(self):
        s = ["Then the stock doubles every day for 3 days."]
        self.assertIsNone(math_solver.solve(self.q(7, s), ""))


class Data(unittest.TestCase):
    d = os.path.join(FIX, "data_analysis_0002")
    df = pd.read_csv(os.path.join(d, "sales.csv"))

    def s(self, q): return data_solver.solve(PRE + q + POST, self.d)

    def test_cases(self):
        df = self.df
        self.assertEqual(self.s("How many units of product 'widget' were sold in months 7 through 8 inclusive?"),
                         str(df[(df["product"] == "widget") & df["month"].between(7, 8)]["units"].sum()))
        self.assertEqual(self.s("What is the total revenue (units * price) in region 'east'?"),
                         str(int((df[df.region == "east"].units * df[df.region == "east"].price).sum())))
        self.assertEqual(self.s("How many orders were placed in region 'west'?"), str((df.region == "west").sum()))
        self.assertEqual(self.s("How many units were sold in month 12?"), str(df[df.month == 12].units.sum()))
        self.assertEqual(self.s("What is the average price of product 'widget'? Round to 2 decimals."),
                         f"{df[df['product'] == 'widget'].price.mean():.2f}")
        self.assertEqual(self.s("What is the maximum units in region 'south'?"), str(df[df.region == "south"].units.max()))

    def test_missing_file_is_infeasible(self):
        self.assertEqual(data_solver.solve("The file sales2.csv has columns a,b. How many units?", self.d), UNSOLVABLE)

    def test_unsure_float_returns_none(self):
        self.assertIsNone(self.s("What is the average price of product 'widget'?"))

    def test_median(self):
        df = self.df
        self.assertEqual(self.s("What is the median price of product 'widget'?"),
                         str(int(df[df["product"] == "widget"].price.median())))

    def test_distinct_counts(self):
        df = self.df
        self.assertEqual(self.s("How many unique products were sold in region 'west'?"),
                         str(df[df.region == "west"]["product"].nunique()))
        self.assertEqual(self.s("How many distinct regions are there?"), str(df.region.nunique()))

    def test_percentage_of_orders(self):
        df = self.df
        self.assertEqual(self.s("What percentage of orders were placed in region 'east'?"),
                         str(round(100 * len(df[df.region == "east"]) / len(df))))

    def test_groupby_top(self):
        df = self.df
        by_units = df.groupby("product").units.sum()
        self.assertEqual(self.s("Which product had the highest total units?"), by_units.idxmax())
        by_count = df.groupby("region").size()
        self.assertEqual(self.s("Which region sold the most orders?"), by_count.idxmax())
        by_avg = df.groupby("region").price.mean()
        self.assertEqual(self.s("Which region had the lowest average price?"), by_avg.idxmin())

    def test_groupby_unknown_measure_returns_none(self):
        self.assertIsNone(self.s("Which product had the highest color?"))

    def test_groupby_tie_returns_none(self):
        d = tempfile.mkdtemp(prefix="data_tie_")
        try:
            with open(os.path.join(d, "t.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["group", "value"])
                w.writerows([["a", 1], ["b", 1], ["a", 2], ["b", 2]])
            q = "The file t.csv has columns group,value (4 rows). Which group has the highest total value?"
            self.assertIsNone(data_solver.solve(q, d))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Search(unittest.TestCase):
    d = os.path.join(FIX, "search_synth")
    P = "The directory kb/ contains a small knowledge base of markdown files. "
    T = " Use the file tools or bash (ls, grep, cat) to find it. Work only inside the current directory."

    def s(self, q): return search_solver.solve(self.P + q + self.T, self.d)

    def test_cases(self):
        self.assertEqual(self.s("In which city is the headquarters of Alice Smith's employer?"), "Osaka")
        self.assertEqual(self.s("In which year was Alice Novak born?"), "1981")
        self.assertEqual(self.s("Who is the mentor of Alice Smith?"), "Olga Kowalski")
        self.assertEqual(self.s("In which year was the employer of Alice Novak founded?"), "2010")

    def test_missing_entity_is_infeasible(self):
        self.assertEqual(self.s("In which year was Zed Quark born?"), UNSOLVABLE)

    def test_broken_chain_falls_back(self):
        # mentor has no card -> cannot continue the chain; never guess
        self.assertIsNone(self.s("In which city is the headquarters of the employer of the mentor of Alice Smith?"))


class SearchSynonyms(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="kb_syn_")
        kb = os.path.join(self.d, "kb")
        os.makedirs(kb)
        docs = {
            "person_ada.md": "# Ada Lovelace\nBorn: 1815\nEmployer: Analytical Engine Ltd\n"
                             "Spouse: William King\nCountry: UK\nMentor: Charles Babbage\n",
            "company_analytical.md": "# Analytical Engine Ltd\nHeadquarters: London\nFounded: 1822\n"
                                     "Founder: Charles Babbage\nCEO: Ada Lovelace\nParent: Babbage Holdings\n",
            "company_babbage.md": "# Babbage Holdings\nHeadquarters: Cambridge\nCEO: Charles Babbage\n",
            "book_notes.md": "# Notes on the Engine\nAuthor: Ada Lovelace\nDirector: Ada Lovelace\n",
        }
        for name, text in docs.items():
            with open(os.path.join(kb, name), "w", encoding="utf-8") as f:
                f.write(text)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def s(self, q):
        return search_solver.solve(q + " Work only inside the current directory.", self.d)

    def test_where_located(self):
        self.assertEqual(self.s("In which city is Ada Lovelace's employer located?"), "London")
        self.assertEqual(self.s("Where is Analytical Engine Ltd headquartered?"), "London")

    def test_works_at(self):
        self.assertEqual(self.s("Which company does Ada Lovelace work at?"), "Analytical Engine Ltd")
        self.assertEqual(self.s("Which country is Ada Lovelace a citizen of?"), "UK")

    def test_founder_ceo_parent(self):
        self.assertEqual(self.s("Who is the founder of Analytical Engine Ltd?"), "Charles Babbage")
        self.assertEqual(self.s("Who is the CEO of Analytical Engine Ltd?"), "Ada Lovelace")
        self.assertEqual(self.s("Who is the parent company of Analytical Engine Ltd?"), "Babbage Holdings")

    def test_author_spouse_director(self):
        self.assertEqual(self.s("Who wrote Notes on the Engine?"), "Ada Lovelace")
        self.assertEqual(self.s("Who is Ada Lovelace married to?"), "William King")
        self.assertEqual(self.s("Who directed Notes on the Engine?"), "Ada Lovelace")

    def test_when_founded_is_not_confused_with_founder(self):
        self.assertEqual(self.s("When was Analytical Engine Ltd founded?"), "1822")




# ---------------- math v2 + code repair ----------------
import json, random, subprocess
from solvers import code_solver, math_solver as _ms


class Math2(unittest.TestCase):
    def test_sample_template_2(self):
        q = ("A warehouse starts with 1669 boxes. It ships 36 boxes per day for 3 days. "
             "It receives 2 deliveries of 87 boxes each. How many boxes does it have at the end?")
        self.assertEqual(_ms.solve(q, ""), "1735")

    def test_mixed_variants(self):
        q = ("A shop starts with 500 items. Then it sells 20 items per day for 4 days. Then it receives 100 items. "
             "Then it loses 30 items. How many items does it have at the end?")
        self.assertEqual(_ms.solve(q, ""), str(500 - 80 + 100 - 30))

    def test_unknown_step(self):
        self.assertIsNone(_ms.solve("A warehouse starts with 5 boxes. It is robbed twice. How many boxes?", ""))


BASE = {
 "reverse_words": "def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
 "total": "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
 "maximum": "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
 "minimum": "def minimum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
 "average": "def average(xs):\n    return sum(xs) / len(xs)\n",
 "first_n": "def first_n(xs, n):\n    return xs[:n]\n",
 "count_even": "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
 "count_odd": "def count_odd(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
}
TESTS = """import unittest
import mylib as m
class T(unittest.TestCase):
    def test_reverse_words(self): self.assertEqual(m.reverse_words('c a b'), 'b a c')
    def test_total(self): self.assertEqual(m.total([1, 2, 3]), 6); self.assertEqual(m.total([5]), 5)
    def test_maximum(self): self.assertEqual(m.maximum([3, 9, 2]), 9); self.assertEqual(m.maximum([9, 1]), 9)
    def test_minimum(self): self.assertEqual(m.minimum([3, 9, 2]), 2); self.assertEqual(m.minimum([1, 5]), 1)
    def test_average(self): self.assertEqual(m.average([2, 4, 6]), 4); self.assertAlmostEqual(m.average([1, 2, 4]), 7 / 3)
    def test_first_n(self): self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7]); self.assertEqual(m.first_n([1, 2], 1), [1])
    def test_count_even(self): self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)
    def test_count_odd(self): self.assertEqual(m.count_odd([1, 2, 4, 6, 7]), 2)
if __name__ == '__main__': unittest.main()
"""
BUGS = [  # (function, old, new)
 ("reverse_words", "reversed", "sorted"), ("total", "s += x", "s -= x"), ("total", "s = 0", "s = 1"),
 ("maximum", "x > best", "x < best"), ("maximum", "xs[0]", "xs[1]"), ("minimum", "x < best", "x > best"),
 ("average", "len(xs)", "(len(xs) + 1)"), ("average", " / ", " // "), ("first_n", "xs[:n]", "xs[:n - 1]"),
 ("count_even", "== 0", "== 1"), ("count_odd", "== 1", "== 0"),
]
INSTR = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py "
         "(do not edit the tests) until all tests pass.")


def _build(d, bugs):
    os.makedirs(d, exist_ok=True)
    parts = dict(BASE)
    for fn, old, new in bugs:
        parts[fn] = parts[fn].replace(old, new, 1)
    open(f"{d}/mylib.py", "w").write("\n\n".join(parts.values()))
    open(f"{d}/test_mylib.py", "w").write(TESTS)


class CodeRepair(unittest.TestCase):
    def check(self, bugs, tag):
        d = f"/tmp/cr_{tag}"; shutil.rmtree(d, ignore_errors=True); _build(d, bugs)
        self.assertEqual(code_solver.solve(INSTR, d), "DONE", bugs)
        p = subprocess.run([sys.executable, "test_mylib.py"], cwd=d, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr[-300:])
        ns = {}; exec(open(f"{d}/mylib.py").read(), ns)  # behaviour must match references on random inputs
        rnd = random.Random(1)
        for name, ref in code_solver.REF.items():
            for args in code_solver._inputs(name, rnd):
                self.assertEqual(ns[name](*args), ref(*args), (name, args))

    def test_each_single_bug(self):
        for i, b in enumerate(BUGS):
            with self.subTest(bug=b):
                self.check([b], f"s{i}")

    def test_two_bugs_at_once(self):
        self.check([BUGS[0], BUGS[3]], "d1")
        self.check([BUGS[1], BUGS[8], BUGS[9]], "d3")

    def test_untouched_tests_file(self):
        d = "/tmp/cr_keep"; shutil.rmtree(d, ignore_errors=True); _build(d, [BUGS[0]])
        before = open(f"{d}/test_mylib.py").read(); code_solver.solve(INSTR, d)
        self.assertEqual(before, open(f"{d}/test_mylib.py").read())

    def test_not_fixable_returns_none(self):
        d = "/tmp/cr_hard"; os.makedirs(d, exist_ok=True)
        open(f"{d}/mylib.py", "w").write("def f():\n    return 'wrong'\n")
        open(f"{d}/test_mylib.py", "w").write("import unittest, mylib as m\nclass T(unittest.TestCase):\n    def test_f(self): self.assertEqual(m.f(), 'right')\nif __name__=='__main__': unittest.main()\n")
        self.assertIsNone(code_solver.solve(INSTR, d))


if __name__ == "__main__":
    unittest.main(verbosity=2)
