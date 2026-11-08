from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI


class TaskSolver:
    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.client = OpenAI(
            base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.neuraldeep.ru/v1"),
            api_key=api_key or os.getenv("LLM_API_KEY", "demo-key"),
        )

    def solve(self, task: dict, model: str) -> dict:
        prompt = str(task.get("prompt", ""))
        response = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = response.choices[0].message.content or ""
        return {"status": "ok", "answer": answer}

    def save_answer(self, task: dict, content: str, output_dir: str | Path) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        answer_file = out / "answer.txt"
        answer_file.write_text(content, encoding="utf-8")
        return answer_file
