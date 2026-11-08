import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import planner

FIX = os.path.join(ROOT, "tests/fixtures/tasks")


def mk(name, typ, reward, instr="x" * 200):
    return {"id": name, "dir": os.path.join(FIX, "math_0000"), "type": typ, "reward": reward,
            "instruction": instr, "bytes": 0}


class T(unittest.TestCase):
    def test_orders_by_value_per_token(self):
        rows = planner.rank([(mk("cheap_hi", "math", 5), 1.0), (mk("dear_lo", "coding", 1), 1.0),
                             (mk("cheap_lo", "math", 1), 1.0)], 100000)
        ids = [r["task"]["id"] for r in rows]
        self.assertEqual(ids[0], "cheap_hi"); self.assertEqual(ids[-1], "dear_lo")

    def test_reserve_and_infeasibility(self):
        items = [(mk(f"t{i}", "coding", 3), 1.0) for i in range(10)]
        est = planner.est_cost(planner.features(items[0][0]))
        rows = planner.rank(items, est * 10, reserve=0.15)
        self.assertEqual(sum(r["selected"] for r in rows), 8)  # 85% of 10 tasks' worth
        low = planner.rank([(mk("a", "math", 3), 1.0), (mk("b", "math", 3), 0.1)], 100000)
        self.assertEqual(low[0]["task"]["id"], "a")

    def test_est_in_plausible_range(self):
        for typ in ("math", "data_analysis", "search", "coding"):
            e = planner.est_cost(planner.features(mk("x", typ, 1)))
            self.assertTrue(100 < e < 1500, (typ, e))


if __name__ == "__main__":
    unittest.main()
