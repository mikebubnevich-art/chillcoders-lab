from __future__ import annotations

from project.config.settings import MODEL_CATALOG


class ModelRouter:
    def __init__(self, model_catalog: dict | None = None) -> None:
        self.model_catalog = model_catalog or MODEL_CATALOG

    def choose_model(self, task: dict, remaining_tokens: int) -> str:
        reward = float(task.get("reward", 0))
        task_text = str(task.get("prompt", "") + " " + task.get("description", ""))
        text_size = len(task_text)

        if reward >= 20 and text_size > 2000 and remaining_tokens > 20000:
            return "qwen3.6-35b-a3b"
        if reward >= 10 and text_size > 1000:
            return "qwen3.8-27b"
        if remaining_tokens < 15000:
            return "qwen3.8-27b-noreason"
        if reward < 5:
            return "qwen3.8-27b-noreason"
        return "qwen3.6-35b-a3b-noreason"
