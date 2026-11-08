"""Persist results after EVERY solved task (unsaved == unsolved in the rules).

Two outputs:
  1. <task_dir>/answer.txt   (required by the rules)
  2. OUT file with {task_id: answer}. If OUT is a directory (or has no suffix) we
     write OUT/solutions.json. If OUT is not writable we fall back to /tmp.
"""
import json
import os
import threading


class Results:
    def __init__(self, out_env: str | None):
        self.path = self._resolve(out_env)
        self.data = {}
        self._lock = threading.Lock()

    @staticmethod
    def _resolve(out):
        if not out:
            return "/tmp/solutions.json"
        if os.path.isdir(out) or not os.path.splitext(out)[1]:
            try:
                os.makedirs(out, exist_ok=True)
            except OSError:
                return "/tmp/solutions.json"
            return os.path.join(out, "solutions.json")
        return out

    def save(self, task, answer: str) -> None:
        ans = answer.strip()
        with open(os.path.join(task["dir"], "answer.txt"), "w", encoding="utf-8") as f:
            f.write(ans + "\n")
        with self._lock:
            self.data[task["id"]] = ans
            self._flush()

    def _flush(self):
        for p in (self.path, "/tmp/solutions.json"):
            try:
                tmp = p + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=1)
                os.replace(tmp, p)
                return
            except OSError:
                continue
