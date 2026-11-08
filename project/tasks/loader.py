from __future__ import annotations

import json
from pathlib import Path


class TaskLoader:
    def __init__(self, tasks_dir: str | Path) -> None:
        self.tasks_dir = Path(tasks_dir)

    def load_tasks(self) -> list[dict]:
        tasks: list[dict] = []
        for task_dir in sorted(self.tasks_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_file = task_dir / "task.json"
            if not task_file.exists():
                continue
            with task_file.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            payload.setdefault("task_id", task_dir.name)
            payload.setdefault("reward", 0)
            payload.setdefault("estimated_cost", 5000)
            payload.setdefault("prompt", payload.get("description", ""))
            tasks.append(payload)
        return tasks
