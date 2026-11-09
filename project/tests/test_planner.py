import json, os, sys, tempfile, unittest
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

    def test_empirical_stats_refine_cost_and_success(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "traj.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for cost, reward, ok in [(120, 5, True), (130, 5, True), (200, 5, False)]:
                    f.write(json.dumps({"type": "math", "cost": cost, "reward": reward,
                                        "success": ok, "status": "solved" if ok else "failed"}) + "\n")
            stats = planner.load_trajectory_stats(path)
            self.assertIn("math", stats)
            self.assertAlmostEqual(stats["math"]["avg_cost"], 150.0, places=1)
            self.assertGreaterEqual(stats["math"]["p90_cost"], 150.0)
            self.assertGreater(stats["math"]["success_rate"], 0.5)
            f = planner.features(mk("emp", "math", 5))
            # conservative: a cheap empirical bucket never lowers the prior
            self.assertGreaterEqual(planner.est_cost(f, stats=stats), planner.est_cost(f))

    def test_expensive_empirical_bucket_raises_the_estimate(self):
        f = planner.features(mk("emp", "math", 5))
        raw = planner.est_cost(f)
        stats = {"math": {"avg_cost": 900, "p90_cost": 5000, "success_rate": 0.9, "count": 10}}
        self.assertGreater(planner.est_cost(f, stats=stats), raw)
        self.assertGreater(planner.p_success(f, 1.0, stats=stats), planner.p_success(f))

    def test_single_sample_barely_moves_the_prior(self):
        f = planner.features(mk("one", "coding", 3))
        raw = planner.est_cost(f)
        stats = {"coding": {"avg_cost": 9000, "p90_cost": 9000, "success_rate": 1.0, "count": 1}}
        self.assertLess(planner.est_cost(f, stats=stats) - raw, 0.2 * raw)

    def test_load_planner_stats_from_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "planner_stats.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"type": "coding", "cost": 900, "success": True}) + "\n")
                f.write(json.dumps({"type": "coding", "cost": 1100, "success": False}) + "\n")
            stats = planner.load_planner_stats(path)
            self.assertIn("coding", stats)
            self.assertAlmostEqual(stats["coding"]["avg_cost"], 1000.0, places=1)
            self.assertAlmostEqual(stats["coding"]["success_rate"], 0.5, places=2)

    def test_client_writes_task_level_planner_rows(self):
        from harness.budget import Budget
        from harness.llm_client import LLMClient
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "traj.jsonl")
            c = LLMClient(Budget(1000), traj_path=path)
            c.log_task({"task": "t1", "type": "math", "cost": 42.5,
                        "success": True, "src": "llm0"})
            c._log({"tag": "t1#0", "model": "m", "usage": {}, "cost": 1})
            stats = planner.load_trajectory_stats(path)
            self.assertEqual(stats["math"]["count"], 1)  # per-call rows are not tasks
            self.assertAlmostEqual(stats["math"]["avg_cost"], 42.5)
            self.assertEqual(stats["math"]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
