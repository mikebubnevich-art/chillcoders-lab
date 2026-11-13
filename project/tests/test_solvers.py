import csv, os, shutil, subprocess, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from solvers import UNSOLVABLE, code_solver, data_solver, logic_solver, math_solver, search_solver, shell_solver, try_solve

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

    def test_tie_without_rule_is_unsolvable(self):
        # Two groups share the top value: the question has no unique answer, so
        # the task is provably not answerable as posed (skip instead of guessing).
        d = tempfile.mkdtemp(prefix="data_tie_")
        try:
            with open(os.path.join(d, "t.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["group", "value"])
                w.writerows([["a", 1], ["b", 1], ["a", 2], ["b", 2]])
            q = "The file t.csv has columns group,value (4 rows). Which group has the highest total value?"
            self.assertEqual(data_solver.solve(q, d), UNSOLVABLE)
            q2 = q + " If several groups tie, answer with the alphabetically first."
            self.assertEqual(data_solver.solve(q2, d), "a")
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
        self.assertEqual(self.s("Which city is the company where Ada Lovelace works based in?"), "London")

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


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(FIX))))
GEN = os.path.join(REPO, "generated_tasks")


class GeneratedBenchmark(unittest.TestCase):
    """The committed generated_tasks/ set is the deterministic regression corpus."""

    def gen(self, tid):
        import json
        d = os.path.join(GEN, tid)
        meta = json.load(open(os.path.join(d, "task.json"), encoding="utf-8"))
        return {"id": tid, "dir": d, "root": d, "type": meta["task_type"],
                "instruction": meta["instruction"], "reward": meta.get("reward", 1)}

    def test_feasible_generated_tasks_are_solved_deterministically(self):
        expected = {"data_easy_0003": "50", "data_medium_0004": "154",
                    "math_easy_0001": "274", "math_hard_0002": "268", "math_medium_0009": "47",
                    "search_easy_0005": "1987", "search_medium_0012": "2004",
                    "shell_medium_0008": "92"}
        for tid, want in expected.items():
            with self.subTest(tid=tid):
                self.assertEqual(str(try_solve(self.gen(tid), dry_run=True)), want)

    def test_unsolvable_generated_tasks_get_the_literal_answer(self):
        from harness.router import route_task
        for tid in ("unsolvable_data_0011", "unsolvable_logic_0010"):
            with self.subTest(tid=tid):
                r = route_task(self.gen(tid))
                self.assertEqual(r.status, "solved")
                self.assertEqual(r.answer, "UNSOLVABLE")

    def test_generated_coding_tasks_are_fixed_and_use_the_pass_marker(self):
        for tid in ("coding_medium_0006", "coding_hard_0007"):
            with self.subTest(tid=tid):
                d = tempfile.mkdtemp(prefix=f"gen_{tid}_")
                try:
                    shutil.copytree(os.path.join(GEN, tid), d, dirs_exist_ok=True)
                    t = self.gen(tid)
                    self.assertEqual(code_solver.solve(t["instruction"], d), "PASS")
                    p = subprocess.run([sys.executable, "-m", "unittest", "test_mylib.py"], cwd=d,
                                       capture_output=True, text=True)
                    self.assertEqual(p.returncode, 0, p.stderr[-400:])
                finally:
                    shutil.rmtree(d, ignore_errors=True)


class FixtureHonesty(unittest.TestCase):
    """A coding fixture must fail *before* repair, or a green gate proves nothing.

    Both task-dir and generated_tasks once contained already-repaired modules,
    because an earlier solver run wrote its result back into the committed copy.
    """

    def _fails_at_start(self, root):
        ws = os.path.join(root, "workspace")
        work = ws if os.path.isdir(ws) else root
        test = next(f for f in sorted(os.listdir(work)) if f.startswith("test_") and f.endswith(".py"))
        d = tempfile.mkdtemp(prefix="honest_")
        try:
            shutil.copytree(work, d, dirs_exist_ok=True)
            p = subprocess.run([sys.executable, test], cwd=d, capture_output=True, text=True, timeout=30)
            return p.returncode
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_generated_coding_tasks_fail_before_repair(self):
        for tid in ("coding_medium_0006", "coding_hard_0007"):
            with self.subTest(tid=tid):
                self.assertNotEqual(self._fails_at_start(os.path.join(GEN, tid)), 0)

    def test_sample_coding_tasks_fail_before_repair(self):
        # только оригинальные сэмплы; наши копии тех же задач живут в generated_tasks
        base = os.path.join(REPO, "task-dir", "task-dir")
        for tid in ("coding_0003", "coding_0007"):
            with self.subTest(tid=tid):
                self.assertNotEqual(self._fails_at_start(os.path.join(base, tid)), 0)


class Markers(unittest.TestCase):
    def test_answer_marker(self):
        from harness.storage import answer_marker, normalize_coding_answer
        self.assertEqual(answer_marker("Then write only PASS to answer.txt."), "PASS")
        self.assertEqual(answer_marker("write ONLY the word DONE to the file answer.txt"), "DONE")
        self.assertEqual(answer_marker("no marker here"), "DONE")
        self.assertEqual(normalize_coding_answer("write exactly PASS to answer.txt", "DONE"), "PASS")
        self.assertEqual(normalize_coding_answer("write exactly PASS to answer.txt", "PASS"), "PASS")
        self.assertEqual(normalize_coding_answer("no marker", "DONE"), "DONE")


class Math3(unittest.TestCase):
    def test_has_with_lots_and_floor_bundles(self):
        q = ("A shop has 36 boxes with 24 pencils each. It sells 175 pencils, then receives 8 boxes "
             "with 15 pencils each. The remaining pencils are packed into bundles of 17; incomplete "
             "bundles are not counted. How many complete bundles can be made?")
        self.assertEqual(_ms.solve(q, ""), "47")

    def test_fraction_and_discarded_remainder(self):
        q = ("A tank contains 987 liters. First remove one third of the water (round the amount "
             "removed down to a whole liter). Add 125 liters. Then divide the water equally among 4 "
             "tanks and keep one share, discarding any remainder. Finally add 73 liters. How many "
             "liters remain?")
        self.assertEqual(_ms.solve(q, ""), "268")

    def test_non_exact_division_without_hint_is_unknown(self):
        q = ("A shop has 10 items. Then divide the items equally among 3 boxes and keep one share. "
             "How many items remain?")
        self.assertIsNone(_ms.solve(q, ""))


class MathShadow(unittest.TestCase):
    """Families from the hard-but-honest corpus (tools/gen_shadow.py)."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="math_shadow_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_ships_half_of_remaining(self):
        q = ("A warehouse starts with 720 boxes. Then it ships out half of the remaining boxes. "
             "Then it receives 2 deliveries of 37 boxes each. Then it throws away 20 damaged "
             "boxes. How many boxes does it have at the end?")
        self.assertEqual(_ms.solve(q, self.d), "414")

    def test_repack_reworded_bundle(self):
        # 'repacked' (no word boundary before 'packed') must still match BUNDLE
        q = ("A depot receives 8 crates of flour, each holding 2500 grams. The flour is repacked "
             "into bags of 600 grams. How many full bags can be filled?")
        self.assertEqual(_ms.solve(q, self.d), "33")

    def test_start_value_from_csv(self):
        with open(os.path.join(self.d, "inventory.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item", "stock"])
            w.writerow(["bolts", 805])
            w.writerow(["nuts", 300])
        q = ("A warehouse starts with the number of bolts listed in inventory.csv. Then it ships "
             "out 4 orders of 48 bolts each. Then it receives 49 more bolts. How many bolts are "
             "left?")
        self.assertEqual(_ms.solve(q, self.d), "662")

    def test_start_value_from_json_key(self):
        with open(os.path.join(self.d, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"start_stock": 884, "other": 1}, f)
        q = ("A warehouse starts with the stock given in config.json (key start_stock). It ships "
             "out 4 orders of 36 boxes each, then receives 45 boxes. How many boxes are left?")
        self.assertEqual(_ms.solve(q, self.d), "785")

    def test_start_value_from_missing_file_is_unknown(self):
        q = ("A warehouse starts with the stock given in config.json (key start_stock). Then it "
             "receives 5 boxes. How many boxes are there?")
        self.assertIsNone(_ms.solve(q, self.d))

    def test_percent_discount_then_tax(self):
        q = ("A jacket costs $653. It is discounted by 10%, then 8% sales tax is added to the "
             "discounted price. What is the final price? Round to 2 decimals.")
        # 653*0.9=587.7 ; *1.08=634.716 -> half-up 634.72
        self.assertEqual(_ms.solve(q, self.d), "634.72")

    def test_costs_start_is_matched(self):
        q = ("A jacket costs $400. It is discounted by 25%, then 10% sales tax is added to the "
             "discounted price. What is the final price? Round to 2 decimals.")
        self.assertEqual(_ms.solve(q, self.d), "330.00")


class Data2(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="data_v3_")
        with open(os.path.join(self.d, "orders.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "city", "quantity", "price", "month"])
            w.writerows([[1, "Oslo", 10, 4, 1], [2, "Riga", 5, 8, 2], [3, "Oslo", 7, 9, 2],
                         [4, "Oslo", 8, 5, 4], [5, "Oslo", 3, 9, 4]])
        with open(os.path.join(self.d, "t.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["category", "quarter", "units", "unit_price"])
            w.writerows([["pen", 1, 14, 2], ["pen", 2, 21, 2], ["pen", 2, 13, 3], ["pen", 2, 16, 2]])
        with open(os.path.join(self.d, "emp.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["employee_id", "department"])
            w.writerows([["E1", "Sales"], ["E2", "Support"]])

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_quantity_price_revenue_without_question_mark(self):
        q = ("Using orders.csv, calculate the total revenue (quantity multiplied by price for each "
             "row) for orders from Oslo in months 2 through 4 inclusive. Use python3.")
        self.assertEqual(data_solver.solve(q, self.d), "130")

    def test_total_number_of_units_not_revenue(self):
        q = ("The file t.csv contains sales transactions. Find the total number of units of pen sold "
             "in quarter 2 (sum units, not revenue). Use python3 to calculate it.")
        self.assertEqual(data_solver.solve(q, self.d), "50")

    def test_numeric_measure_missing_is_unsolvable(self):
        q = "The file emp.csv has columns employee_id,department. What was the total salary?"
        self.assertEqual(data_solver.solve(q, self.d), UNSOLVABLE)


class ShellDeterministicRegression(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="shell_reg_")
        with open(os.path.join(self.d, "data.txt"), "w", encoding="utf-8") as f:
            f.write("1\n2\n3\n4\n5\n6\n")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_even_filter_is_respected_for_average(self):
        q = "What is the average of the even numbers in data.txt?"
        self.assertEqual(shell_solver.solve(q, self.d), "4")

    def test_odd_filter_is_respected_for_median(self):
        q = "What is the median of the odd numbers in data.txt?"
        self.assertEqual(shell_solver.solve(q, self.d), "3")

    def test_threshold_filter_is_respected_for_sum(self):
        q = "What is the sum of numbers greater than 3 in data.txt?"
        self.assertEqual(shell_solver.solve(q, self.d), "15")


class LogicDeterministicRegression(unittest.TestCase):
    def test_partition_exceeds_total_is_unsolvable(self):
        q = "There are 10 fruits. 4 are apples and 3 are oranges. The rest are pears. There are 4 pears."
        self.assertEqual(logic_solver.solve(q, ""), UNSOLVABLE)

    def test_no_other_and_missing_remaining_is_unsolvable(self):
        q = "There are 10 fruits. 4 are apples and 3 are oranges. No other fruits exist."
        self.assertEqual(logic_solver.solve(q, ""), UNSOLVABLE)

    def test_remaining_can_close_the_total(self):
        q = "There are 10 fruits. 4 are apples and 3 are oranges. The rest are pears. The rest are 3."
        self.assertIsNone(logic_solver.solve(q, ""))


class DataComparisonFilters(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="data_filters_")
        with open(os.path.join(self.d, "sales.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["order_id", "region", "product", "month", "units", "price"])
            w.writerows([
                [1, "east", "widget", 1, 12, 10],
                [2, "east", "widget", 2, 8, 10],
                [3, "west", "gadget", 1, 14, 7],
                [4, "east", "widget", 2, 11, 8],
            ])

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_numeric_columns_support_generic_greater_than_and_between(self):
        q = "The file sales.csv has columns order_id,region,product,month,units,price (4 rows). How many rows have units greater than 10? Work only inside the current directory."
        self.assertEqual(data_solver.solve(q, self.d), "3")
        q2 = "The file sales.csv has columns order_id,region,product,month,units,price (4 rows). How many rows have price between 7 and 10? Work only inside the current directory."
        self.assertEqual(data_solver.solve(q2, self.d), "4")

    def test_numeric_columns_support_generic_less_than_and_sum(self):
        q = "The file sales.csv has columns order_id,region,product,month,units,price (4 rows). What is the sum of price for rows where price <= 8? Work only inside the current directory."
        self.assertEqual(data_solver.solve(q, self.d), "15")


class SearchProse(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="kb_prose_")
        kb = os.path.join(self.d, "kb")
        os.makedirs(kb)
        with open(os.path.join(kb, "people.md"), "w", encoding="utf-8") as f:
            f.write("# Staff biographies\nMara Chen was born in 1991.\nJonas Reed was born in 1987.\n")
        with open(os.path.join(kb, "company.md"), "w", encoding="utf-8") as f:
            f.write("# Cedarline Systems\nFounded in 2004. Main office: Halifax.\n")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_person_fact_from_prose(self):
        self.assertEqual(search_solver.solve("In which year was Jonas Reed born?", self.d), "1987")

    def test_title_fact_from_prose(self):
        self.assertEqual(search_solver.solve("When was Cedarline Systems founded?", self.d), "2004")

    def test_based_in_phrase_is_not_treated_as_two_fields(self):
        kb = os.path.join(self.d, "kb")
        with open(os.path.join(kb, "alice_smith.md"), "w", encoding="utf-8") as f:
            f.write("# Alice Smith\nEmployer: Nova Labs\n")
        with open(os.path.join(kb, "nova_labs.md"), "w", encoding="utf-8") as f:
            f.write("# Nova Labs\nHeadquarters: Osaka\n")
        self.assertEqual(search_solver.solve("Which city is the company where Alice Smith works based in?", self.d), "Osaka")


class ShadowRegressions(unittest.TestCase):
    """Defects the hard-but-honest corpus (tools/gen_shadow.py) exposed."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="shadow_reg_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def csv(self, rows):
        with open(os.path.join(self.d, "sales.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.HDR)
            w.writerows(rows)

    def q(self, text, n):
        return (f"The file sales.csv has columns {','.join(self.hdr)} ({n} rows). {text}"
                " Use bash with python3. Work only inside the current directory.")

    HDR = ["order_id", "region", "product", "month", "units", "price"]
    hdr = HDR

    def test_month_names_range(self):
        self.csv([[1, "north", "widget", "Jul", 5, 10],
                  [2, "north", "widget", "Aug", 7, 10],
                  [3, "north", "widget", "Sep", 9, 10],
                  [4, "north", "gadget", "Jul", 100, 10]])
        q = self.q("How many units of product 'widget' were sold in months July through August inclusive?", 4)
        self.assertEqual(data_solver.solve(q, self.d), "12")

    def test_at_least_threshold(self):
        self.csv([[1, "north", "widget", 1, 30, 10],
                  [2, "north", "widget", 1, 19, 10],
                  [3, "north", "widget", 1, 20, 10]])
        q = self.q("How many orders have at least 20 units?", 3)
        self.assertEqual(data_solver.solve(q, self.d), "2")

    def test_labels_with_trailing_spaces(self):
        self.csv([[1, "north", "widget ", 1, 5, 10],
                  [2, "north", "widget", 1, 7, 10],
                  [3, "north", "gadget", 1, 9, 10]])
        q = self.q("Some labels have trailing spaces. How many units of product 'widget' were sold?", 3)
        self.assertEqual(data_solver.solve(q, self.d), "12")

    def test_half_up_rounding(self):
        rows = [[k, "north" if k < 13 else "west", "widget", 1, 1, 10] for k in range(80)]
        self.csv(rows)
        q = self.q("What percentage of orders were placed in region 'north'? Round to 1 decimal.", 80)
        self.assertEqual(data_solver.solve(q, self.d), "16.3")  # 16.25, not banker's 16.2

    def test_missing_named_column_is_infeasible(self):
        self.csv([[1, "north", "widget", 1, 5, 10]])
        q = self.q("What is the sum of the revenue column?", 1)
        self.assertEqual(data_solver.solve(q, self.d), UNSOLVABLE)

    def test_row_count_contradiction_is_infeasible(self):
        self.csv([[1, "north", "widget", 1, 5, 10]])
        q = self.q("How many orders were placed in region 'north'?", 40)  # file has 1 row
        self.assertEqual(data_solver.solve(q, self.d), UNSOLVABLE)

    def test_header_only_file_is_infeasible(self):
        self.csv([])
        q = self.q("What is the average price in region 'north'? Round to 2 decimals.", 0)
        self.assertEqual(data_solver.solve(q, self.d), UNSOLVABLE)


class SearchNormalization(unittest.TestCase):
    """Titles that differ from the question only by spacing must still resolve."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="kb_norm_")
        self.kb = os.path.join(self.d, "kb")
        os.makedirs(self.kb)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def card(self, fn, text):
        with open(os.path.join(self.kb, fn), "w", encoding="utf-8") as f:
            f.write(text)

    def test_double_space_in_title(self):
        self.card("marta.md", "# Marta  Ruiz\nSpouse: Alice Moreau\n")
        self.assertEqual(search_solver.solve("Who is the spouse of Marta Ruiz?", self.d), "Alice Moreau")

    def test_duplicate_normalized_titles_are_infeasible(self):
        self.card("a.md", "# John Smith\nEmployer: Solstice Foods\n")
        self.card("b.md", "# John  Smith\nEmployer: Kaido Robotics\n")
        self.assertEqual(search_solver.solve("Who is the employer of John Smith?", self.d), UNSOLVABLE)


class SearchReverse(unittest.TestCase):
    """Reverse lookup: the start entity is described by a field value, not named."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="kb_rev_")
        self.kb = os.path.join(self.d, "kb")
        os.makedirs(self.kb)
        self.card("p1.md", "# Jonas Reed\nEmployer: Northwind Traders\nBorn: 1988, Lyon\n")
        self.card("p2.md", "# Kira Petrov\nEmployer: Solstice Foods\nBorn: 1971, Oslo\n")
        self.card("c1.md", "# Northwind Traders\nHeadquarters: Rotterdam\nFounded: 2001\n")
        self.card("c2.md", "# Solstice Foods\nHeadquarters: Oslo\nFounded: 1998\n")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def card(self, fn, text):
        with open(os.path.join(self.kb, fn), "w", encoding="utf-8") as f:
            f.write(text)

    def test_reverse_one_hop(self):
        q = ("The directory kb/ contains a small knowledge base of markdown files. "
             "Which company employs the person who was born in Lyon? "
             "Answer with the company name.")
        self.assertEqual(search_solver.solve(q, self.d), "Northwind Traders")

    def test_reverse_missing_entity_is_infeasible(self):
        q = ("The directory kb/ contains a small knowledge base of markdown files. "
             "Which company employs the person who was born in Prague?")
        self.assertEqual(search_solver.solve(q, self.d), UNSOLVABLE)

    def test_reverse_multiple_candidates_no_unique_answer(self):
        self.card("p3.md", "# Irina Sato\nEmployer: Cobalt Works\nBorn: 1990, Lyon\n")
        q = ("The directory kb/ contains a small knowledge base of markdown files. "
             "Which company employs the person who was born in Lyon?")
        self.assertIsNone(search_solver.solve(q, self.d))


class MoneyFormats(unittest.TestCase):
    """The numeric detector: currencies, EU/CH grouping, accounting negatives, percents."""

    def test_dec_formats(self):
        cases = {
            "$1,204.50": "1204.50", "€1.204,50": "1204.50", "1 204,50": "1204.50",
            "1'204.50": "1204.50", "1204.50 USD": "1204.50", "₽1204,50": "1204.50",
            "USD 1 204.50": "1204.50", "(1,204.50)": "-1204.50", "-1,204.50": "-1204.50",
            "20%": "20", "1,204": "1204", "1204.505": "1204.505", "1.204": "1204",
            "": None, "abc": None, "1 204 50": None,
        }
        for raw, want in cases.items():
            got = data_solver._dec(raw)
            self.assertEqual(str(got) if got is not None else None, want, raw)

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="money_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def _run(self, price, disc, region, extra=""):
        import csv as csvmod
        with open(os.path.join(self.d, "sales.csv"), "w", newline="", encoding="utf-8") as f:
            w = csvmod.writer(f)
            w.writerow(["order_id", "region", "product", "units", "price", "discount"])
            w.writerow([1, "north", "widget", 2, price, disc])
            w.writerow([2, region, "widget", 3, price, disc])
            w.writerow([3, "south", "gadget", 9, "10", ""])
        q = (f"The file sales.csv has columns order_id,region,product,units,price,discount (3 rows). "
             f"Prices are written like \"{price}\"; discount is a percent and may be empty{extra}. "
             f"What is the total revenue in region '{region}', where revenue is units * price "
             f"with the discount applied? Round to 2 decimals.")
        return data_solver.solve(q, self.d)

    def test_usd_with_discount(self):
        # (2 + 3) * 1204.50 * 0.9 = 5420.25
        self.assertEqual(self._run("$1,204.50", "10", "north"), "5420.25")

    def test_eu_space_with_pct_discount(self):
        # (2 + 3) * 1204.50 * 0.85 = 5119.125 -> half-up 5119.13
        self.assertEqual(self._run("1 204,50", "15%", "north"), "5119.13")

    def test_no_discount_word_means_plain_revenue(self):
        import csv as csvmod
        with open(os.path.join(self.d, "sales.csv"), "w", newline="", encoding="utf-8") as f:
            w = csvmod.writer(f)
            w.writerow(["order_id", "region", "product", "units", "price", "discount"])
            w.writerow([1, "north", "widget", 2, "$1,204.50", ""])
            w.writerow([2, "south", "gadget", 1, "10", ""])
            w.writerow([3, "south", "gadget", 1, "10", ""])
        q = ("The file sales.csv has columns order_id,region,product,units,price,discount (3 rows). "
             "What is the total revenue in region 'north'? Round to 2 decimals.")
        self.assertEqual(data_solver.solve(q, self.d), "2409.00")


class CodeBodyReplacement(unittest.TestCase):
    """Defects outside the single-mutation grammar are repaired from known bodies."""

    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="code_body_")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def write(self, src, test):
        with open(os.path.join(self.d, "mylib.py"), "w", encoding="utf-8") as f:
            f.write(src)
        with open(os.path.join(self.d, "test_mylib.py"), "w", encoding="utf-8") as f:
            f.write(test)

    def run_tests(self):
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
        return subprocess.run([sys.executable, "test_mylib.py"], cwd=self.d, capture_output=True,
                              text=True, env=env)

    def test_wrong_bound_order_is_fixed(self):
        src = "def clamp(value, low, high):\n    return min(low, max(value, high))\n"
        test = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
                "    def test_inside(self):\n        self.assertEqual(mylib.clamp(5, 0, 10), 5)\n"
                "    def test_high(self):\n        self.assertEqual(mylib.clamp(42, 0, 10), 10)\n\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n")
        self.write(src, test)
        self.assertNotEqual(self.run_tests().returncode, 0)  # honest start
        marker = code_solver.solve("Fix the bugs in `mylib.py`, then run `python3 test_mylib.py`. "
                                   "Write only PASS to answer.txt.", self.d)
        self.assertEqual(marker, "PASS")
        self.assertEqual(self.run_tests().returncode, 0)

    def test_missing_function_is_not_declared_infeasible(self):
        """Tests call a name the module never had: adding code is allowed, so
        this must go to the LLM fallback (None), never to UNSOLVABLE."""
        src = "def add(a, b):\n    return a + b\n"
        test = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(mylib.add(1, 2), 3)\n"
                "    def test_missing(self):\n        self.assertEqual(mylib.solve_all([1, 2]), 3)\n\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n")
        self.write(src, test)
        marker = code_solver.solve("Run `python3 test_mylib.py`; some tests fail. Fix the bugs in "
                                   "mylib.py (do not edit the tests) until all tests pass. Write "
                                   "only DONE to answer.txt.", self.d)
        self.assertIsNone(marker)
        self.assertNotEqual(self.run_tests().returncode, 0)  # the module must stay untouched

    def test_missing_reset_is_fixed(self):
        src = ("def encode(text):\n"
               "    if not text:\n        return ''\n"
               "    parts, count = [], 1\n"
               "    for i in range(1, len(text)):\n"
               "        if text[i] == text[i - 1]:\n            count += 1\n"
               "        else:\n            parts.append(text[i - 1] + str(count))\n"
               "    if count > 1:\n        parts.append(text[-1] + str(count))\n"
               "    return ''.join(parts)\n")
        test = ("import unittest\nimport mylib\n\n\nclass T(unittest.TestCase):\n"
                "    def test_basic(self):\n        self.assertEqual(mylib.encode('aaabbc'), 'a3b2c1')\n"
                "    def test_single(self):\n        self.assertEqual(mylib.encode('abc'), 'a1b1c1')\n\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n")
        self.write(src, test)
        self.assertNotEqual(self.run_tests().returncode, 0)
        self.assertEqual(code_solver.solve(
            "Fix the bugs in `mylib.py`, then run `python3 test_mylib.py`. "
            "Write only DONE to answer.txt.", self.d), "DONE")
        self.assertEqual(self.run_tests().returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
