import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import dsl

FIX = os.path.join(ROOT, "tests/fixtures/tasks")


def task(name):
    d = os.path.join(FIX, name)
    with open(os.path.join(d, "task.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return {"id": meta["task_id"], "dir": d, "type": meta["task_type"],
            "instruction": meta["instruction"], "reward": meta.get("reward", 1)}


def plan(obj):
    return "```json\n" + json.dumps(obj) + "\n```"


class Math(unittest.TestCase):
    t0 = task("math_0000")
    t4 = task("math_0004")

    def test_chain_template(self):
        p = {"start": 1752, "ops": [
            {"op": "div", "k": 2, "round": "down"},
            {"op": "mul", "k": 2},
            {"op": "div", "k": 3, "round": "down"},
            {"op": "mul", "k": 2},
            {"op": "div", "k": 3, "round": "down"}]}
        self.assertEqual(dsl.execute(self.t0, plan(p)), ("389", None))

    def test_day_and_lots_template(self):
        p = {"start": 1669, "ops": [{"op": "sub", "amount": 36, "times": 3},
                                    {"op": "add", "lots": 2, "amount": 87}]}
        self.assertEqual(dsl.execute(self.t4, plan(p)), ("1735", None))

    def test_exact_fraction(self):
        p = {"start": 1, "ops": [{"op": "div", "k": 8, "round": "exact"}]}
        self.assertEqual(dsl.execute(self.t4, plan(p)), ("0.125", None))

    def test_rounding(self):
        p = {"start": 10, "ops": [{"op": "div", "k": 3, "round": "down"}]}
        self.assertEqual(dsl.execute(self.t4, plan(p)), ("3", None))
        p = {"start": 10, "ops": [{"op": "div", "k": 3, "round": "up"}]}
        self.assertEqual(dsl.execute(self.t4, plan(p)), ("4", None))

    def test_inexact_division_is_rejected(self):
        ans, err = dsl.execute(self.t4, plan({"start": 10, "ops": [{"op": "div", "k": 3}]}))
        self.assertIsNone(ans)
        self.assertIn("round", err)

    def test_unknown_key_is_rejected(self):
        ans, err = dsl.execute(self.t4, plan({"start": 10, "ops": [{"op": "sub", "amount": 3, "count": 2}]}))
        self.assertIsNone(ans)
        self.assertIn("unknown keys", err)

    def test_bad_json_falls_back_with_error(self):
        ans, err = dsl.execute(self.t4, "I think the answer is 42")
        self.assertIsNone(ans)
        self.assertIn("JSON", err)

    def test_bare_json_without_fence(self):
        ans, _ = dsl.execute(self.t4, json.dumps({"start": 5, "ops": [{"op": "add", "amount": 2}]}))
        self.assertEqual(ans, "7")


class Data(unittest.TestCase):
    t = task("data_analysis_0002")

    def test_sum_template(self):
        p = {"filters": [{"col": "product", "op": "eq", "value": "widget"},
                         {"col": "month", "op": "between", "lo": 7, "hi": 8}],
             "agg": "sum", "col": "units"}
        self.assertEqual(dsl.execute(self.t, plan(p)), ("158", None))

    def test_avg_with_decimals(self):
        p = {"filters": [{"col": "product", "op": "eq", "value": "widget"}],
             "agg": "avg", "col": "price"}
        ans, err = dsl.execute(self.t, plan(p))
        self.assertIsNone(ans)  # no format hint -> never guess
        self.assertIn("decimals", err)
        self.assertEqual(dsl.execute(self.t, plan({**p, "decimals": 2}))[0], "42.22")

    def test_count_and_count_distinct(self):
        self.assertEqual(dsl.execute(self.t, plan({"agg": "count"}))[0], "20")
        p = {"filters": [{"col": "region", "op": "eq", "value": "west"}],
             "agg": "count_distinct", "col": "product"}
        self.assertEqual(dsl.execute(self.t, plan(p))[0], "3")

    def test_median(self):
        p = {"filters": [{"col": "product", "op": "eq", "value": "widget"}],
             "agg": "median", "col": "price"}
        self.assertEqual(dsl.execute(self.t, plan(p))[0], "33")

    def test_group_top(self):
        p = {"agg": "group_top", "group": "product", "col": "units", "mode": "sum"}
        self.assertEqual(dsl.execute(self.t, plan(p))[0], "widget")
        p = {"agg": "group_top", "group": "region", "mode": "count"}
        self.assertEqual(dsl.execute(self.t, plan(p))[0], "west")
        p = {"agg": "group_top", "group": "region", "col": "price", "mode": "avg", "direction": "min"}
        self.assertEqual(dsl.execute(self.t, plan(p))[0], "south")

    def test_percent_of_orders(self):
        p = {"filters": [{"col": "region", "op": "eq", "value": "west"}],
             "agg": "percent", "decimals": 1}
        self.assertEqual(dsl.execute(self.t, plan(p))[0], "35.0")
        # without filters the share is 100%
        self.assertEqual(dsl.execute(self.t, plan({"agg": "percent"}))[0], "100")

    def test_sum_product_revenue(self):
        p = {"filters": [{"col": "region", "op": "eq", "value": "west"}],
             "agg": "sum_product", "cols": ["units", "price"]}
        self.assertEqual(dsl.execute(self.t, plan(p))[0], "12144")
        self.assertIsNone(dsl.execute(self.t, plan({"agg": "sum_product", "cols": ["units"]}))[0])

    def test_unknown_column_and_key(self):
        self.assertIsNone(dsl.execute(self.t, plan({"agg": "sum", "col": "nope"}))[0])
        p = {"filters": [{"col": "region", "op": "eq", "value": "west", "typo": 1}],
             "agg": "count"}
        self.assertIsNone(dsl.execute(self.t, plan(p))[0])

    def test_in_and_contains_filters(self):
        p = {"filters": [{"col": "region", "op": "in", "values": ["east", "south"]}],
             "agg": "count"}
        self.assertEqual(dsl.execute(self.t, plan(p))[0], "9")
        p = {"filters": [{"col": "product", "op": "contains", "value": "wid"}],
             "agg": "count"}
        self.assertEqual(dsl.execute(self.t, plan(p))[0], "9")

    def test_no_rows_sum_is_zero(self):
        p = {"filters": [{"col": "region", "op": "eq", "value": "nowhere"}],
             "agg": "sum", "col": "units"}
        self.assertEqual(dsl.execute(self.t, plan(p))[0], "0")

    def test_group_tie_is_rejected(self):
        d = tempfile.mkdtemp(prefix="dsl_tie_")
        try:
            with open(os.path.join(d, "t.csv"), "w", encoding="utf-8") as f:
                f.write("group,value\na,1\nb,1\na,2\nb,2\n")
            t = {"dir": d, "type": "data_analysis",
                 "instruction": "The file t.csv has columns group,value. Which group has the highest total value?"}
            p = {"agg": "group_top", "group": "group", "col": "value", "mode": "sum"}
            ans, err = dsl.execute(t, plan(p))
            self.assertIsNone(ans)
            self.assertIn("tie", err)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Search(unittest.TestCase):
    t = task("search_synth")

    def test_sample(self):
        self.assertEqual(dsl.execute(self.t, plan({"start": "Alice Smith",
                                                   "hops": ["employer", "headquarters"]})), ("Osaka", None))

    def test_synonym_hop(self):
        self.assertEqual(dsl.execute(self.t, plan({"start": "Alice Smith", "hops": ["employer", "city"]}))[0],
                         "Osaka")

    def test_missing_entity_and_bad_hop(self):
        self.assertIsNone(dsl.execute(self.t, plan({"start": "Zed Quark", "hops": ["born"]}))[0])
        self.assertIsNone(dsl.execute(self.t, plan({"start": "Alice Smith", "hops": ["employer", "nope"]}))[0])

    def test_intermediate_value_must_be_an_entity(self):
        ans, err = dsl.execute(self.t, plan({"start": "Alice Smith", "hops": ["born", "headquarters"]}))
        self.assertIsNone(ans)
        self.assertIn("not a known entity", err)

    def test_ambiguous_synonym_is_rejected(self):
        d = tempfile.mkdtemp(prefix="dsl_kb_")
        try:
            kb = os.path.join(d, "kb")
            os.makedirs(kb)
            with open(os.path.join(kb, "a.md"), "w", encoding="utf-8") as f:
                f.write("# Acme\nHeadquarters: Oslo\nLocation: Bergen\n")
            t = {"dir": d, "type": "search", "instruction": "Where is Acme based in?"}
            ans, err = dsl.execute(t, plan({"start": "Acme", "hops": ["based in"]}))
            self.assertIsNone(ans)
            self.assertIn("ambiguous", err)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class Specs(unittest.TestCase):
    def test_spec_only_for_supported_types(self):
        self.assertIsNotNone(dsl.spec(task("math_0000")))
        self.assertIsNotNone(dsl.spec(task("data_analysis_0002")))
        self.assertIsNotNone(dsl.spec(task("search_synth")))
        self.assertIsNone(dsl.spec(task("coding_0007")))
        self.assertIsNone(dsl.spec({"type": "unknown", "dir": FIX, "instruction": ""}))

    def test_data_context_has_columns(self):
        s = dsl.spec(task("data_analysis_0002"))
        self.assertIn("sales.csv", s.context)
        self.assertIn("product", s.context)
        self.assertIn("units", s.context)

    def test_search_context_has_fields(self):
        s = dsl.spec(task("search_synth"))
        self.assertIn("employer", s.context)
        self.assertIn("headquarters", s.context)

    def test_math_context_is_empty(self):
        self.assertEqual(dsl.spec(task("math_0000")).context, "")

    def test_context_bug_does_not_break_spec(self):
        orig = dsl._csv_schema
        dsl._csv_schema = lambda _task: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            s = dsl.spec(task("data_analysis_0002"))
        finally:
            dsl._csv_schema = orig
        self.assertIsNotNone(s)
        self.assertEqual(s.context, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
