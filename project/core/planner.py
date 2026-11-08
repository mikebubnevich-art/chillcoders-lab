from __future__ import annotations

from dataclasses import dataclass, field

from project.core.budget import BudgetManager
from project.core.router import ModelRouter


@dataclass
class Planner:
    budget: BudgetManager
    router: ModelRouter = field(default_factory=ModelRouter)

    def should_skip(self, task: dict) -> bool:
        reward = float(task.get("reward", 0))
        estimated_cost = float(task.get("estimated_cost", 0))
        expected_value = reward / max(estimated_cost, 1)
        return expected_value < 1.0 or not self.budget.can_afford(int(estimated_cost))

    def select_task(self, tasks: list[dict]) -> dict | None:
        candidates = [task for task in tasks if not self.should_skip(task)]
        if not candidates:
            return None
        return max(candidates, key=lambda task: float(task.get("reward", 0)) / max(float(task.get("estimated_cost", 1)), 1))

    def decide_model(self, task: dict) -> str:
        return self.router.choose_model(task, self.budget.remaining_tokens or 0)

    def run(self, tasks: list[dict]) -> list[dict]:
        results = []
        while self.budget.remaining_tokens and self.budget.remaining_tokens > 0:
            next_task = self.select_task(tasks)
            if next_task is None:
                break
            next_task["model"] = self.decide_model(next_task)
            results.append(next_task)
            tasks.remove(next_task)
        return results
