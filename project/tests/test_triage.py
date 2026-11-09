import json, os, shutil, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from harness.tasks import load_tasks
from harness.triage import triage
from harness.router import route_task
from harness import feasibility
from solvers import try_solve
import gen_infeasible

FIX = os.path.join(ROOT, "tests/fixtures/tasks")


class T(unittest.TestCase):
    def test_feasible_fixtures_are_never_flagged(self):
        d = "/tmp/tri_ok"; shutil.rmtree(d, ignore_errors=True); shutil.copytree(FIX, d)
        for t in load_tasks(d):
            r = triage(t)
            self.assertEqual(r.status, "solved", (t["id"], r.reasons))

    def test_feasible_fixtures_not_flagged_without_solvers(self):
        d = "/tmp/tri_ok2"; shutil.rmtree(d, ignore_errors=True); shutil.copytree(FIX, d)
        for t in load_tasks(d):
            self.assertEqual(triage(t, use_solvers=False).status, "llm", t["id"])  # heuristics alone: no false positives

    def test_all_synthetic_infeasible_are_caught_for_free(self):
        out = "/tmp/tri_bad"; made = gen_infeasible.make(FIX, out)
        self.assertGreaterEqual(len(made), 12)
        missed = []
        for t in load_tasks(out):
            if triage(t).status != "infeasible":
                missed.append(t["id"])
        self.assertEqual(missed, [])

    def test_heuristics_alone_catch_most(self):
        out = "/tmp/tri_bad2"; made = gen_infeasible.make(FIX, out)
        caught = sum(triage(t, use_solvers=False).status == "infeasible" for t in load_tasks(out))
        self.assertGreaterEqual(caught / len(made), 0.8)

    def test_solvers_match_expected_answers(self):
        for t in load_tasks(FIX):
            if t["type"] not in ("math", "data_analysis", "search"):
                continue
            with open(os.path.join(t["dir"], "task.json"), encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(try_solve(t), meta["expected"], t["id"])

    def test_dry_run_solver_never_writes_into_task_dir(self):
        d = "/tmp/tri_dry"; shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(os.path.join(FIX, "coding_0007"), d)
        before = open(os.path.join(d, "mylib.py"), "rb").read()
        t = {"id": "coding_0007", "type": "coding", "dir": d, "reward": 2,
             "instruction": "Fix bugs in `mylib.py`. Run `python3 test_mylib.py`."}
        r = triage(t, use_solvers=False)
        self.assertEqual(r.status, "llm")
        self.assertEqual(open(os.path.join(d, "mylib.py"), "rb").read(), before)

    def test_output_file_need_not_exist(self):
        t = {"dir": os.path.join(FIX, "math_0000"),
             "instruction": "A warehouse has 10 boxes. Compute the answer and write it to results.csv."}
        r = feasibility.check(t)
        self.assertFalse(r.hard, r.reasons)
        self.assertIn("output_file:results.csv", r.reasons)

    def test_router_is_the_single_source_of_truth(self):
        t = next(tt for tt in load_tasks(FIX) if tt["type"] == "math")
        r = route_task(t)
        self.assertEqual(r.status, "solved")
        self.assertEqual(r.route, "solver")
        self.assertEqual(triage(t).status, r.status)

    def test_missing_input_file_is_hard(self):
        t = {"dir": os.path.join(FIX, "math_0000"),
             "instruction": "Use the starting stock from inventory.csv. How many boxes are left?"}
        r = feasibility.check(t)
        self.assertTrue(r.hard, r.reasons)
        self.assertIn("missing_file:inventory.csv", r.reasons)


if __name__ == "__main__":
    unittest.main(verbosity=2)
