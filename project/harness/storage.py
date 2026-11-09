"""Persist results after EVERY solved task (unsaved == unsolved in the rules).

Two outputs:
  1. <task_dir>/answer.txt   (required by the rules)
  2. OUT file with {task_id: answer}. If OUT is a directory (or has no suffix) we
     write OUT/solutions.json. If OUT is not writable we fall back to /tmp.
"""
import json
import os
import re
import threading

MARKER_RE = re.compile(
    r"write\s+(?:exactly\s+|only\s+)?(?:the\s+word\s+)?([A-Z]{2,})\s+to\s+"
    r"(?:the\s+file\s+)?answer\.txt", re.I)
CODING_TYPES = ("coding", "code")


def requested_marker(instruction: str) -> str:
    """The exact marker the instruction asks for, or '' when it names none."""
    m = MARKER_RE.search(instruction or "")
    return m.group(1).upper() if m else ""


def answer_marker(instruction: str) -> str:
    """Marker to verify/store for a coding task (DONE is the historical default)."""
    return requested_marker(instruction) or "DONE"


def normalize_coding_answer(instruction: str, answer: str) -> str:
    """'DONE' and the explicitly requested marker are interchangeable.

    Models often obey the old DONE convention even when the task asks for
    PASS; the checker wants the marker written in the instruction.
    """
    want = requested_marker(instruction)
    ans = str(answer).strip()
    if want and ans.upper() in ("DONE", want):
        return want
    return ans


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
        if task.get("type") in CODING_TYPES:
            ans = normalize_coding_answer(task.get("instruction", ""), ans)
        # The checker may read answer.txt in the working dir or in the task
        # dir; write both when they differ (workspace layout).
        targets = [os.path.join(task["dir"], "answer.txt")]
        root = task.get("root")
        if root and os.path.abspath(root) != os.path.abspath(task["dir"]):
            targets.append(os.path.join(root, "answer.txt"))
        saved, last_error = False, None
        for path in targets:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(ans + "\n")
                saved = True
            except OSError as e:
                last_error = e
        if not saved and last_error is not None:
            raise last_error
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
