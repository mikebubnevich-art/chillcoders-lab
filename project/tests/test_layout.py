import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from harness.storage import Results
from harness.tasks import load_tasks
from harness.triage import triage

FIX = os.path.join(ROOT, "tests/fixtures/tasks")


def make_task(base, workspace=True, files=None):
    """Build <tmp>/<id>/task.json + workspace/ (or flat) from a fixture."""
    d = tempfile.mkdtemp(prefix="layout_")
    name = "task_0001"
    task_root = os.path.join(d, name)
    work = os.path.join(task_root, "workspace") if workspace else task_root
    os.makedirs(work)
    with open(os.path.join(base, "task.json"), encoding="utf-8") as f:
        meta = json.load(f)
    meta["task_id"] = name
    with open(os.path.join(task_root, "task.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    for rel in files or os.listdir(base):
        if rel == "task.json":
            continue
        src = os.path.join(base, rel)
        dst = os.path.join(work, rel)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    return d, name, task_root, work


class Layout(unittest.TestCase):
    def test_workspace_is_the_working_dir(self):
        d, name, task_root, work = make_task(os.path.join(FIX, "data_analysis_0002"))
        try:
            t = load_tasks(d)[0]
            self.assertEqual(t["dir"], work)
            self.assertEqual(t["root"], task_root)
            self.assertEqual(triage(t).status, "solved")
            self.assertEqual(triage(t).answer, "158")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_task_json_inside_workspace_is_tolerated(self):
        d, name, task_root, work = make_task(os.path.join(FIX, "data_analysis_0002"))
        try:
            os.replace(os.path.join(task_root, "task.json"),
                       os.path.join(work, "task.json"))
            t = load_tasks(d)[0]
            self.assertEqual(t["dir"], work)
            self.assertEqual(t["root"], task_root)
            self.assertEqual(triage(t).answer, "158")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_flat_layout_is_unchanged(self):
        d, name, task_root, work = make_task(os.path.join(FIX, "data_analysis_0002"),
                                             workspace=False)
        try:
            t = load_tasks(d)[0]
            self.assertEqual(t["dir"], task_root)
            self.assertEqual(t["root"], task_root)
            self.assertEqual(triage(t).status, "solved")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_search_and_coding_workspace(self):
        for base, expected in (("search_synth", "Osaka"), ("coding_0007", None)):
            with self.subTest(base=base):
                d, _, _, work = make_task(os.path.join(FIX, base))
                try:
                    t = load_tasks(d)[0]
                    if expected is None:
                        self.assertEqual(triage(t).status, "solved")
                    else:
                        self.assertEqual(triage(t).answer, expected)
                finally:
                    shutil.rmtree(d, ignore_errors=True)

    def test_results_writes_answer_in_both_locations(self):
        d, name, task_root, work = make_task(os.path.join(FIX, "data_analysis_0002"))
        out = tempfile.mkdtemp(prefix="layout_out_")
        try:
            t = load_tasks(d)[0]
            Results(out).save(t, "158")
            for p in (os.path.join(work, "answer.txt"), os.path.join(task_root, "answer.txt")):
                with open(p, encoding="utf-8") as f:
                    self.assertEqual(f.read().strip(), "158")
        finally:
            shutil.rmtree(d, ignore_errors=True)
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
