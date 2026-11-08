"""Compact task context: listing + tiny samples. Never ships whole data files."""
import os

SKIP = {"task.json", "answer.txt"}


def build_context(task_dir: str, budget_chars: int = 5000) -> str:
    entries = []
    for root, dirs, files in os.walk(task_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith((".", "__")))
        depth = root[len(task_dir):].count(os.sep)
        if depth > 2:
            dirs[:] = []
            continue
        for fn in sorted(files):
            if fn in SKIP or fn.startswith("."):
                continue
            p = os.path.join(root, fn)
            entries.append((os.path.relpath(p, task_dir), os.path.getsize(p)))
    lines = [f"{rel} ({sz}B)" for rel, sz in entries[:40]]
    if len(entries) > 40:
        lines.append(f"... +{len(entries) - 40} more files")
    out = ["Files:"] + lines
    used = 0
    for rel, sz in entries:
        if used > budget_chars:
            break
        p = os.path.join(task_dir, rel)
        low = rel.lower()
        try:
            if low.endswith(".csv"):
                with open(p, encoding="utf-8", errors="replace") as f:
                    head = [next(f, "") for _ in range(4)]
                snippet = f"--- {rel} head (4 lines of {sz}B) ---\n" + "".join(head)
            elif low.endswith(".py"):
                with open(p, encoding="utf-8", errors="replace") as f:
                    snippet = f"--- {rel} ---\n" + f.read(3000)
            elif low.endswith(".json"):
                with open(p, encoding="utf-8", errors="replace") as f:
                    snippet = f"--- {rel} head ---\n" + f.read(400)
            else:
                continue  # md/txt: listed only, the script greps them
        except OSError:
            continue
        used += len(snippet)
        out.append(snippet)
    return "\n".join(out)
