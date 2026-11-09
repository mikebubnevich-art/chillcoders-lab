import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import runner


class T(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="sbx_test_")
        with open(os.path.join(self.d, "data.txt"), "w", encoding="utf-8") as f:
            f.write("hello")
        with open(os.path.join(self.d, "mylib.py"), "w", encoding="utf-8") as f:
            f.write("def f():\n    return 1\n")
        with open(os.path.join(self.d, "test_x.py"), "w", encoding="utf-8") as f:
            f.write("import unittest\n\nclass T(unittest.TestCase):\n    def test_f(self):\n        self.assertEqual(1, 1)\n")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def sb(self, timeout=5):
        return runner.Sandbox(self.d, timeout=timeout)

    def test_script_writes_stay_in_the_sandbox(self):
        sb = self.sb()
        try:
            rc, _ = sb.run("open('data.txt','w').write('changed')\n"
                           "open('answer.txt','w').write('42')")
            self.assertEqual(rc, 0)
            self.assertEqual(open(os.path.join(self.d, "data.txt")).read(), "hello")
            self.assertEqual(sb.read_answer(), "42")
            self.assertFalse(os.path.exists(os.path.join(self.d, "answer.txt")))
            sb.commit(("answer.txt",))
            self.assertEqual(open(os.path.join(self.d, "answer.txt")).read(), "42")
        finally:
            sb.cleanup()

    def test_timeout_returns_124(self):
        sb = self.sb(timeout=1)
        try:
            rc, out = sb.run("while True:\n    pass")
            self.assertEqual(rc, 124, out)
        finally:
            sb.cleanup()

    def test_tests_are_restored_after_tampering_and_deletion(self):
        sb = self.sb()
        try:
            sb.run("open('test_x.py','w').write('broken')\n"
                   "import os\nos.remove('test_x.py')")
            self.assertTrue(sb.restore_tests())
            restored = open(os.path.join(sb.dir, "test_x.py"), encoding="utf-8").read()
            self.assertIn("class T", restored)
        finally:
            sb.cleanup()

    def test_changed_python_excludes_tests(self):
        sb = self.sb()
        try:
            sb.run("open('mylib.py','w').write('def f():\\n    return 2\\n')\n"
                   "open('test_x.py','w').write('bad')")
            sb.restore_tests()
            changed = sb.changed_python()
            self.assertIn("mylib.py", changed)
            self.assertNotIn("test_x.py", changed)
        finally:
            sb.cleanup()

    def test_in_place_fallback_still_restores_tests(self):
        old = runner.MAX_COPY_MB
        runner.MAX_COPY_MB = 0  # force the >limit fallback
        try:
            sb = self.sb()
            try:
                self.assertFalse(sb.isolated)
                sb.run("open('test_x.py','w').write('broken')")
                self.assertTrue(sb.restore_tests())  # restored from the in-memory snapshot
                self.assertIn("class T", open(os.path.join(self.d, "test_x.py"), encoding="utf-8").read())
            finally:
                sb.cleanup()
        finally:
            runner.MAX_COPY_MB = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
