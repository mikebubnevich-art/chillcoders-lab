import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bench_solvers
import gen_synthetic

OUT = "/tmp/synth_bench_test"


class T(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gen_synthetic.make(OUT, seed=11)
        cls.rows, cls.sec = bench_solvers.run(OUT)
        cls.per, cls.ov = bench_solvers.summarize(cls.rows)

    def test_no_wrong_answers(self):
        self.assertEqual(self.ov["wrong"], 0,
                         [r for r in self.rows
                          if r["expected"] is not None and r["status"] == "solved"
                          and r["answer"] != r["expected"]])

    def test_no_false_skips(self):
        self.assertEqual(self.ov["false_skip"], 0)

    def test_coverage(self):
        cov = self.ov["correct"] / max(1, self.ov["feasible_total"])
        self.assertGreaterEqual(cov, 0.95, (cov, self.ov))

    def test_all_infeasible_caught(self):
        self.assertEqual(self.ov["infeasible_caught"], self.ov["infeasible_total"])

    def test_expected_mix(self):
        self.assertGreaterEqual(self.ov["infeasible_total"] / self.ov["total"], 0.15)
        self.assertLessEqual(self.ov["infeasible_total"] / self.ov["total"], 0.25)
        for typ in ("math", "data_analysis", "search", "coding"):
            self.assertGreater(self.per[typ]["feasible_total"], 0, typ)


if __name__ == "__main__":
    unittest.main(verbosity=2)
