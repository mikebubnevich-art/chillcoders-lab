import os, sys, threading, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.budget import Budget, BudgetExceeded
from harness.llm_client import cost_of


class T(unittest.TestCase):
    def test_cost_formula(self):
        self.assertEqual(cost_of({"prompt_tokens": 1000, "completion_tokens": 50}), 150.0)
        # reasoning reported separately and larger than completion -> added
        self.assertEqual(cost_of({"prompt_tokens": 0, "completion_tokens": 10,
                                  "completion_tokens_details": {"reasoning_tokens": 90}}), 100.0)
        # reasoning already inside completion -> not double counted
        self.assertEqual(cost_of({"prompt_tokens": 0, "completion_tokens": 100,
                                  "completion_tokens_details": {"reasoning_tokens": 90}}), 100.0)

    def test_parallel_never_overshoots(self):
        b = Budget(1000, safety=1.0); ok = []
        def w():
            try: b.reserve(100); b.commit(100, 100); ok.append(1)
            except BudgetExceeded: pass
        ts = [threading.Thread(target=w) for _ in range(50)]
        [t.start() for t in ts]; [t.join() for t in ts]
        self.assertEqual(len(ok), 10); self.assertLessEqual(b.spent, 1000)

    def test_waits_for_other_reservations_but_fails_when_truly_exhausted(self):
        b = Budget(1000, safety=1.0); b.reserve(800)
        res = []
        t = threading.Thread(target=lambda: (b.reserve(500, wait=5), res.append("ok")))
        t.start(); threading.Event().wait(0.3)
        self.assertEqual(res, [])          # blocked, not rejected
        b.commit(800, 100)                 # first task really cost only 100
        t.join(3); self.assertEqual(res, ["ok"])
        b2 = Budget(1000, safety=1.0); b2.reserve(100); b2.commit(100, 900)
        with self.assertRaises(BudgetExceeded):
            b2.reserve(500, wait=5)        # spent alone cannot cover it -> immediate failure


if __name__ == "__main__":
    unittest.main()
