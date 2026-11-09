import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import agent

FIX = os.path.join(ROOT, "tests/fixtures/tasks")


class T(unittest.TestCase):
    def test_cap_fits_two_coding_attempts(self):
        cap = agent.task_cap({"type": "coding"})
        self.assertGreaterEqual(cap, 2 * agent.MAX_OUT["coding"])

    def test_reasoning_models_get_larger_caps(self):
        self.assertEqual(agent.out_mult("qwen3.8-27b-noreason"), 1.0)
        self.assertGreater(agent.out_mult("qwen3.8-27b"), 1.0)
        tight = agent.task_cap({"type": "math"})
        wide = agent.task_cap({"type": "math"}, mult=agent.out_mult("qwen3.8-27b"))
        self.assertGreater(wide, tight)

    def test_coding_model_switch(self):
        coding = {"type": "coding"}
        self.assertEqual(agent.model_for(coding, "base", "strong"), "strong")
        self.assertEqual(agent.model_for(coding, "base", "strong", "always"), "strong")
        self.assertEqual(agent.model_for(coding, "base", "strong", "fallback"), "base")
        self.assertEqual(agent.model_for({"type": "math"}, "base", "strong"), "base")
        self.assertEqual(agent.model_for(coding, "base", ""), "base")

    def test_coding_fallback_only_after_a_failed_coding_task(self):
        coding = {"type": "coding"}
        self.assertEqual(agent.fallback_model(coding, "failed", "base", "strong"), "strong")
        self.assertEqual(agent.fallback_model(coding, "failed", "base", "strong", "always"), "")
        self.assertEqual(agent.fallback_model(coding, "budget", "base", "strong"), "")
        self.assertEqual(agent.fallback_model(coding, "failed", "base", ""), "")
        self.assertEqual(agent.fallback_model({"type": "math"}, "failed", "base", "strong"), "")

    def test_test_command_from_instruction(self):
        cmds = agent.test_commands("Run `python3 test_mylib.py` in the current directory.", FIX)
        self.assertEqual(cmds, [[sys.executable, "test_mylib.py"]])

    def test_fallback_to_test_files(self):
        d = os.path.join(FIX, "coding_0007")
        cmds = agent.test_commands("Fix mylib.py until tests pass.", d)
        self.assertTrue(any(c[-1].endswith("test_mylib.py") for c in cmds), cmds)

    def test_test_command_without_backticks(self):
        d = os.path.join(FIX, "coding_0007")
        cmds = agent.test_commands("Run python3 test_mylib.py in the current directory.", d)
        self.assertEqual(cmds, [[sys.executable, "test_mylib.py"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
