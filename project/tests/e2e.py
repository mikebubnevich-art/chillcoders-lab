"""End-to-end check on fixtures with the mock proxy.
  python tests/e2e.py            -> normal run + retry run + tiny-budget run
"""
import json, os, re, shutil, subprocess, sys, time, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests/fixtures/tasks")
EXPECT = {"data_analysis_0002": "158", "math_0000": "389", "math_0004": "1735", "search_synth": "Osaka",
          "coding_0007": "DONE", "coding_variant2": "DONE"}


def run(budget, break_first=False, port=8765, solvers=True, src=FIX, extra_env=None):
    work = "/tmp/e2e_tasks"; out = "/tmp/e2e_out"
    shutil.rmtree(work, ignore_errors=True); shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(src, work)
    extra = extra_env or {}
    srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "tests/mock_proxy.py"), str(port)],
                           env={**os.environ, "MOCK_BREAK_FIRST": "1" if break_first else "0", **extra})
    time.sleep(0.6)
    env = {**os.environ, "OUT": out, "LLM_BASE_URL": f"http://127.0.0.1:{port}/v1",
           "LLM_API_KEY": "test", "BUDGET": str(budget), "TRAJ": "/tmp/e2e_traj.jsonl",
           # dev venv has no full openjiuwen runtime: keep plumbing tests on
           # urllib; the adapter test overrides this with USE_OJIWEN=1.
           "USE_OJIWEN": "0",
           # pin the models: default MODEL_CODING would scale coding caps
           "MODEL": "qwen3.8-27b-noreason", "MODEL_CODING": "",
           "DISABLE_SOLVERS": "0" if solvers else "1", **extra}
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


def _full_openjiuwen():
    import contextlib, io
    try:
        # `import openjiuwen.core.runner` is lazy; Runner pulls pymilvus etc.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            from openjiuwen.core.runner import Runner  # noqa: F401
        return True
    except Exception:
        return False


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
        work, out, log, _ = run(100000, break_first=True, port=8766, solvers=False,
                                extra_env={"DISABLE_DSL": "1"})
        self.assertEqual(answers(work).get("math_0000"), "389")
        self.assertIn("llm1", log)  # solved on the retry

    def test_dsl_solves_math_data_search(self):
        work, out, log, _ = run(100000, port=8773, solvers=False, extra_env={"MOCK_DSL": "1"})
        print(log)
        a = answers(work)
        for k, v in EXPECT.items():
            self.assertEqual(a.get(k), v, k)
        self.assertIn("dsl0", log)
        done = json.loads(log.strip().splitlines()[-1].split("[done] ")[1])
        self.assertEqual(done["solved"], 6)
        self.assertLess(done["spent"], 1000)  # plans are much cheaper than full scripts

    def test_bad_dsl_falls_back_to_script(self):
        work, out, log, _ = run(100000, port=8774, solvers=False,
                                extra_env={"MOCK_DSL": "1", "MOCK_DSL_BAD": "1"})
        print(log)
        a = answers(work)
        for k, v in EXPECT.items():
            self.assertEqual(a.get(k), v, k)
        self.assertIn("llm1", log)  # the script attempt after the failed plan solved them

    def test_tiny_budget_never_overshoots(self):
        work, out, log, _ = run(900, port=8767, solvers=False, extra_env={"DISABLE_DSL": "1"})
        print(log)
        done = json.loads(log.strip().splitlines()[-1].split("[done] ")[1])
        self.assertLessEqual(done["spent"], 900)
        self.assertGreaterEqual(done["solved"], 3)  # tasks now queue for budget instead of failing on reservations
        # Wave 1 must contain ONLY planner-selected tasks.
        selected = {m.group(1) for m in re.finditer(r"^\[plan\] (\S+).*\*$", log, re.M)}
        wave2 = log.find("[wave] 2/")
        wave1_log = log[:wave2] if wave2 != -1 else log
        started = set(re.findall(r"^\[task\] (\S+)", wave1_log, re.M))
        self.assertTrue(started <= selected, (started - selected, selected))

    def test_coding_llm_fix_passes_verifier(self):
        src = "/tmp/e2e_coding_ok"
        shutil.rmtree(src, ignore_errors=True); os.makedirs(src)
        shutil.copytree(f"{FIX}/coding_0007", f"{src}/coding_0007")
        work, out, log, _ = run(100000, port=8771, solvers=False, src=src)
        print(log)
        self.assertEqual(answers(work).get("coding_0007"), "DONE")
        r = subprocess.run([sys.executable, "test_mylib.py"], cwd=f"{work}/coding_0007",
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("llm0", log)

    def test_coding_patch_plan_passes_verifier(self):
        src = "/tmp/e2e_coding_patch"
        shutil.rmtree(src, ignore_errors=True); os.makedirs(src)
        for name in ("coding_0007", "coding_variant2"):
            shutil.copytree(f"{FIX}/{name}", f"{src}/{name}")
        work, out, log, _ = run(100000, port=8775, solvers=False, src=src,
                                extra_env={"MOCK_PATCH": "1"})
        print(log)
        for name in ("coding_0007", "coding_variant2"):
            self.assertEqual(answers(work).get(name), "DONE")
            r = subprocess.run([sys.executable, "test_mylib.py"], cwd=f"{work}/{name}",
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("patch0", log)
        done = json.loads(log.strip().splitlines()[-1].split("[done] ")[1])
        self.assertEqual(done["solved"], 2)

    def test_bad_patch_falls_back_to_script(self):
        src = "/tmp/e2e_coding_badpatch"
        shutil.rmtree(src, ignore_errors=True); os.makedirs(src)
        shutil.copytree(f"{FIX}/coding_0007", f"{src}/coding_0007")
        work, out, log, _ = run(100000, port=8776, solvers=False, src=src,
                                extra_env={"MOCK_PATCH": "1", "MOCK_PATCH_BAD": "1"})
        print(log)
        self.assertEqual(answers(work).get("coding_0007"), "DONE")
        self.assertIn("llm1", log)      # the script attempt after the rejected plan
        self.assertNotIn("patch0", log)

    def test_coding_fallback_retries_with_the_strong_model(self):
        src = "/tmp/e2e_coding_fallback"
        shutil.rmtree(src, ignore_errors=True); os.makedirs(src)
        shutil.copytree(f"{FIX}/coding_0007", f"{src}/coding_0007")
        work, out, log, _ = run(100000, port=8780, solvers=False, src=src,
                                extra_env={"MOCK_PATCH": "1",
                                           "MOCK_FAIL_MODEL": "qwen3.8-27b-noreason",
                                           "MODEL": "qwen3.8-27b-noreason",
                                           "MODEL_CODING": "qwen3.8-27b"})
        print(log)
        self.assertEqual(answers(work).get("coding_0007"), "DONE")
        self.assertIn("coding_model=qwen3.8-27b", log)
        self.assertIn("failed", log)          # base model failed first
        self.assertIn("+fb", log)             # the strong model's retry succeeded
        self.assertNotIn("fb-failed", log)

    def test_planner_stats_from_trajectory_raise_the_estimate(self):
        src = "/tmp/e2e_stats_src"
        shutil.rmtree(src, ignore_errors=True); os.makedirs(src)
        shutil.copytree(f"{FIX}/math_0004", f"{src}/math_0004")
        traj = "/tmp/e2e_stats_traj.jsonl"
        with open(traj, "w", encoding="utf-8") as f:
            for _ in range(10):
                f.write(json.dumps({"type": "math", "cost": 5000, "success": True}) + "\n")
        work, out, log, _ = run(100000, port=8782, solvers=False, src=src,
                                extra_env={"TRAJ": traj})
        print(log)
        self.assertIn("[stats] loaded 1 task types", log)
        m = re.search(r"\[plan\] math_0004\s+est=\s*([\d.]+)", log)
        self.assertIsNotNone(m, log)
        self.assertGreater(float(m.group(1)), 300)  # raw prior is ~221
        self.assertEqual(answers(work).get("math_0004"), "1735")

    def test_workspace_layout_coding_via_patch(self):
        src = "/tmp/e2e_workspace"
        shutil.rmtree(src, ignore_errors=True); os.makedirs(src)
        root = f"{src}/coding_ws"
        os.makedirs(f"{root}/workspace")
        shutil.copy2(f"{FIX}/coding_0007/task.json", root)
        for name in ("mylib.py", "test_mylib.py"):
            shutil.copy2(f"{FIX}/coding_0007/{name}", f"{root}/workspace/{name}")
        work, out, log, _ = run(100000, port=8777, solvers=False, src=src,
                                extra_env={"MOCK_PATCH": "1"})
        print(log)
        self.assertEqual(answers(work).get("coding_ws"), "DONE")
        for p in (f"{work}/coding_ws/answer.txt", f"{work}/coding_ws/workspace/answer.txt"):
            self.assertTrue(os.path.exists(p), p)
        r = subprocess.run([sys.executable, "test_mylib.py"], cwd=f"{work}/coding_ws/workspace",
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("patch0", log)

    def test_coding_retry_fits_task_cap(self):
        src = "/tmp/e2e_coding_cap"
        shutil.rmtree(src, ignore_errors=True); os.makedirs(src)
        shutil.copytree(f"{FIX}/coding_0007", f"{src}/coding_0007")
        work, out, log, _ = run(100000, port=8772, solvers=False, src=src,
                                extra_env={"MOCK_CODING_BREAK_FIRST": "1", "MOCK_USAGE_MAX": "1"})
        print(log)
        done = json.loads(log.strip().splitlines()[-1].split("[done] ")[1])
        self.assertEqual(done["solved"], 1)          # the retry was allowed and fixed the code
        self.assertGreater(done["spent"], 1700)      # two 900-token calls fit under the 2400 cap
        self.assertEqual(answers(work).get("coding_0007"), "DONE")

    def test_coding_done_without_fix_is_rejected(self):
        src = "/tmp/e2e_coding_only"
        shutil.rmtree(src, ignore_errors=True); os.makedirs(src)
        shutil.copytree(f"{FIX}/coding_0007", f"{src}/coding_0007")
        work, out, log, _ = run(100000, port=8769, solvers=False, src=src,
                                extra_env={"MOCK_CODING_NOOP": "1"})
        print(log)
        self.assertEqual(answers(work), {})  # DONE without fixing mylib.py is not a solution
        done = json.loads(log.strip().splitlines()[-1].split("[done] ")[1])
        self.assertEqual((done["solved"], done["failed"]), (0, 1))

    def test_coding_test_tamper_is_rejected(self):
        src = "/tmp/e2e_coding_only2"
        shutil.rmtree(src, ignore_errors=True); os.makedirs(src)
        shutil.copytree(f"{FIX}/coding_0007", f"{src}/coding_0007")
        orig_test = open(f"{FIX}/coding_0007/test_mylib.py", encoding="utf-8").read()
        work, out, log, _ = run(100000, port=8770, solvers=False, src=src,
                                extra_env={"MOCK_CODING_TAMPER": "1"})
        print(log)
        self.assertEqual(answers(work), {})
        after = open(f"{work}/coding_0007/test_mylib.py", encoding="utf-8").read()
        self.assertEqual(after, orig_test)  # tests are restored, never modified in the real dir

    @unittest.skipUnless(_full_openjiuwen(), "full openjiuwen runtime is not installed")
    def test_openjiuwen_transport_is_used(self):
        src = "/tmp/e2e_oj_src"
        shutil.rmtree(src, ignore_errors=True); os.makedirs(src)
        shutil.copytree(f"{FIX}/math_0004", f"{src}/math_0004")
        work, out, log, _ = run(100000, port=8781, solvers=False, src=src,
                                extra_env={"USE_OJIWEN": "1", "DISABLE_DSL": "1"})
        print(log)
        self.assertEqual(answers(work).get("math_0004"), "1735")
        recs = [json.loads(line) for line in open("/tmp/e2e_traj.jsonl", encoding="utf-8")
                if line.strip()]
        calls = [r for r in recs if "tag" in r]  # task-level rows carry no transport
        self.assertTrue(calls)
        self.assertTrue(all(r.get("transport") == "openjiuwen" for r in calls), recs)
        self.assertGreater(calls[0]["usage"].get("prompt_tokens", 0), 0)
        done = json.loads(log.strip().splitlines()[-1].split("[done] ")[1])
        self.assertEqual(done["solved"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
