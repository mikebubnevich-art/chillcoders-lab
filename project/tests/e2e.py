"""End-to-end check on fixtures with the mock proxy.
  python tests/e2e.py            -> normal run + retry run + tiny-budget run
"""
import json, os, shutil, subprocess, sys, time, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests/fixtures/tasks")
EXPECT = {"data_analysis_0002": "158", "math_0000": "389", "math_0004": "1735", "search_synth": "Osaka",
          "coding_0007": "DONE", "coding_variant2": "DONE"}


def run(budget, break_first=False, port=8765, solvers=True, src=FIX):
    work = "/tmp/e2e_tasks"; out = "/tmp/e2e_out"
    shutil.rmtree(work, ignore_errors=True); shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(src, work)
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "tests/mock_proxy.py"), str(port)],
                           env={**os.environ, "MOCK_BREAK_FIRST": "1" if break_first else "0"})
    time.sleep(0.6)
    env = {**os.environ, "OUT": out, "LLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
           "LLM_API_KEY": "test", "BUDGET": str(budget), "TRAJ": "/tmp/e2e_traj.jsonl", "DISABLE_SOLVERS": "0" if solvers else "1"}
    if os.path.exists("/tmp/e2e_traj.jsonl"): os.remove("/tmp/e2e_traj.jsonl")
    t0 = time.time()
    p = subprocess.run([sys.executable, os.path.join(ROOT, "run_agent.py"), "--tasks-dir", work],
                       env=env, capture_output=True, text=True)
    srv.terminate()
    return work, out, p.stdout + p.stderr, time.time() - t0


def answers(work):
    r = {}
    for n in os.listdir(work):
        p = os.path.join(work, n, "answer.txt")
        if os.path.exists(p):
            with open(p) as f:
                r[n] = f.read().strip()
    return r


class E2E(unittest.TestCase):
    def test_normal(self):
        work, out, log, sec = run(100000)
        print(log)
        a = answers(work)
        for k, v in EXPECT.items(): self.assertEqual(a.get(k), v, k)
        for c in ("coding_0007", "coding_variant2"):  # the real tests must pass after the 0-token repair
            r = subprocess.run([sys.executable, "test_mylib.py"], cwd=f"{work}/{c}", capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        sol = json.load(open(f"{out}/solutions.json"))
        self.assertEqual(sol["math_0004"], "1735")
        done = json.loads(log.strip().splitlines()[-1].split("[done] ")[1])
        self.assertEqual((done["solved"], done["solver"], done["llm_solved"]), (6, 6, 0))
        self.assertEqual(done["spent"], 0)  # whole fixture set for ZERO tokens

    def test_infeasible_skipped_for_free(self):
        sys.path.insert(0, os.path.join(ROOT, "tools")); import gen_infeasible
        bad = "/tmp/e2e_bad_src"; gen_infeasible.make(FIX, bad)
        work, out, log, _ = run(100000, port=8768, src=bad)
        done = json.loads(log.strip().splitlines()[-1].split("[done] ")[1])
        self.assertEqual(done["solved"], 0); self.assertEqual(done["spent"], 0)
        self.assertEqual(done["infeasible"], len(os.listdir(bad)))
        self.assertEqual(answers(work), {})  # nothing written for infeasible tasks

    def test_retry_once(self):
        work, out, log, _ = run(100000, break_first=True, port=8766, solvers=False)
        self.assertEqual(answers(work).get("math_0000"), "389")
        self.assertIn("llm1", log)  # solved on the retry

    def test_tiny_budget_never_overshoots(self):
        work, out, log, _ = run(900, port=8767, solvers=False)
        print(log)
        done = json.loads(log.strip().splitlines()[-1].split("[done] ")[1])
        self.assertLessEqual(done["spent"], 900)
        self.assertGreaterEqual(done["solved"], 3)  # tasks now queue for budget instead of failing on reservations


if __name__ == "__main__":
    unittest.main(verbosity=2)
