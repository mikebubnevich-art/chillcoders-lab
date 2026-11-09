import json
import os

HIDDEN_KEYS = ("expected", "check")  # never shown to the model / never used by the agent


def load_tasks(tasks_dir: str):
    tasks = []
    for name in sorted(os.listdir(tasks_dir)):
        d = os.path.join(tasks_dir, name)
        tj = os.path.join(d, "task.json")
        if os.path.isdir(d) and not os.path.isfile(tj):
            # tolerate a variant where task.json sits inside workspace/
            alt = os.path.join(d, "workspace", "task.json")
            if os.path.isfile(alt):
                tj = alt
        if not (os.path.isdir(d) and os.path.isfile(tj)):
            continue
        try:
            with open(tj, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        size = 0
        for root, _, files in os.walk(d):
            for fn in files:
                try:
                    size += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
        # Organizer samples keep the files in <task>/workspace; some sets put
        # them directly in <task>. The working dir is where instructions say
        # "current directory", so normalize it once here.
        root = os.path.abspath(d)
        work = os.path.join(root, "workspace")
        if not os.path.isdir(work):
            work = root
        tasks.append({
            "id": meta.get("task_id", name), "dir": work, "root": root, "dirname": name,
            "type": meta.get("task_type", "unknown"),
            "instruction": meta.get("instruction", ""),
            "reward": float(meta.get("reward", 0) or 0),
            "bytes": size,
        })
    return tasks
