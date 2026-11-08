import os, shutil, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from harness.tasks import load_tasks
from harness.triage import triage
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
