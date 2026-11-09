import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness import patch

FIX = os.path.join(ROOT, "tests/fixtures/tasks")
CODING = os.path.join(FIX, "coding_0007")


def plan(obj):
    return "```json\n" + json.dumps(obj) + "\n```"


def task(d, name="coding_0007"):
    with open(os.path.join(d, "task.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return {"id": meta["task_id"], "dir": d, "type": meta["task_type"],
            "instruction": meta["instruction"], "reward": 1}


class Parse(unittest.TestCase):
    def test_valid_plan(self):
        edits, err = patch.parse(plan({"replacements": [
            {"file": "mylib.py", "old": "if x < best:", "new": "if x > best:"}]}))
        self.assertIsNone(err)
        self.assertEqual(edits, [{"file": "mylib.py", "old": "if x < best:", "new": "if x > best:"}])

    def test_bare_json_and_multiple_files(self):
        edits, err = patch.parse(json.dumps({"replacements": [
            {"file": "pkg/a.py", "old": "a", "new": "b"},
            {"file": "b.py", "old": "x", "new": ""}]}))
        self.assertIsNone(err)
        self.assertEqual(len(edits), 2)
        self.assertEqual(edits[1]["new"], "")

    def test_garbage_returns_error(self):
        edits, err = patch.parse("I could not do it")
        self.assertIsNone(edits)
        self.assertIn("JSON", err)

    def test_test_file_is_rejected(self):
        for name in ("test_mylib.py", "tests/helper.py", "mylib_test.py", "conftest.py"):
            with self.subTest(name=name):
                edits, err = patch.parse(plan({"replacements": [
                    {"file": name, "old": "a", "new": "b"}]}))
                self.assertIsNone(edits)
                self.assertIn("read-only", err)

    def test_escape_and_non_python_are_rejected(self):
        for name in ("../outside.py", "/etc/passwd", "notes.txt", ".git/hook.py"):
            with self.subTest(name=name):
                edits, err = patch.parse(plan({"replacements": [
                    {"file": name, "old": "a", "new": "b"}]}))
                self.assertIsNone(edits)

    def test_strict_schema(self):
        bad = [
            {"replacements": []},
            {"replacement": [{"file": "a.py", "old": "a", "new": "b"}]},
            {"replacements": [{"file": "a.py", "old": "", "new": "b"}]},
            {"replacements": [{"file": "a.py", "old": "a"}]},
            {"replacements": [{"file": "a.py", "old": "a", "new": "b", "line": 1}]},
            {"replacements": "nope"},
        ]
        for obj in bad:
            with self.subTest(obj=obj):
                edits, err = patch.parse(plan(obj))
                self.assertIsNone(edits)
                self.assertTrue(err)


class Apply(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="patch_")
        self.work = os.path.join(self.d, "task")
        shutil.copytree(CODING, self.work)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def read(self, rel="mylib.py"):
        with open(os.path.join(self.work, rel), encoding="utf-8") as f:
            return f.read()

    def test_apply_and_revert(self):
        before = self.read()
        edits, _ = patch.parse(plan({"replacements": [
            {"file": "mylib.py", "old": "if x < best:", "new": "if x > best:"},
            {"file": "mylib.py", "old": "x % 2 == 1", "new": "x % 2 == 0"}]}))
        originals = patch.apply(self.work, edits)
        after = self.read()
        self.assertIn("if x > best:", after)
        self.assertIn("x % 2 == 0", after)
        self.assertEqual(originals, {"mylib.py": before.encode()})
        patch.revert(self.work, originals)
        self.assertEqual(self.read(), before)

    def test_old_must_occur_exactly_once(self):
        for old in ("NOPE", "x"):
            with self.subTest(old=old):
                edits = [{"file": "mylib.py", "old": old, "new": "y"}]
                with self.assertRaises(patch.PatchError):
                    patch.apply(self.work, edits)
                self.assertEqual(self.read(), self.read())  # still readable

    def test_nothing_is_written_when_a_later_edit_is_invalid(self):
        before = self.read()
        edits = [{"file": "mylib.py", "old": "if x < best:", "new": "if x > best:"},
                 {"file": "mylib.py", "old": "NOPE", "new": "y"}]
        with self.assertRaises(patch.PatchError):
            patch.apply(self.work, edits)
        self.assertEqual(self.read(), before)

    def test_missing_file_is_rejected(self):
        with self.assertRaises(patch.PatchError):
            patch.apply(self.work, [{"file": "nope.py", "old": "a", "new": "b"}])

    def test_does_not_touch_tests(self):
        tests_before = self.read("test_mylib.py")
        edits = [{"file": "mylib.py", "old": "x % 2 == 1", "new": "x % 2 == 0"}]
        patch.apply(self.work, edits)
        self.assertEqual(self.read("test_mylib.py"), tests_before)


class Context(unittest.TestCase):
    def test_context_has_module_and_tests(self):
        ctx = patch.context(task(CODING))
        self.assertIn("--- mylib.py ---", ctx)
        self.assertIn("if x < best:", ctx)
        self.assertIn("test_mylib.py (read-only tests)", ctx)

    def test_tests_only_means_no_patch_spec(self):
        d = tempfile.mkdtemp(prefix="patch_tests_")
        try:
            with open(os.path.join(d, "test_x.py"), "w", encoding="utf-8") as f:
                f.write("def test_ok():\n    assert True\n")
            self.assertIsNone(patch.spec({"dir": d, "type": "coding"}))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_context_bug_returns_empty(self):
        orig = patch._rel_files
        patch._rel_files = lambda _d: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            self.assertEqual(patch.context(task(CODING)), "")
            self.assertIsNone(patch.spec(task(CODING)))
        finally:
            patch._rel_files = orig


if __name__ == "__main__":
    unittest.main(verbosity=2)
