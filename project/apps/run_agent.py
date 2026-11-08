from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from project.agents.solver import TaskSolver
from project.config.settings import DEFAULT_BUDGET_TOKENS
from project.core.budget import BudgetManager
from project.core.planner import Planner
from project.tasks.loader import TaskLoader
from project.utils.token_estimator import estimate_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-aware benchmark runner")
    parser.add_argument("--tasks-dir", required=True, help="Directory with task folders")
    parser.add_argument("--out-dir", default="./out", help="Directory to write results")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET_TOKENS, help="Budget in tokens")
    args = parser.parse_args()

    task_loader = TaskLoader(args.tasks_dir)
    tasks = task_loader.load_tasks()

    for task in tasks:
        task["estimated_cost"] = max(task.get("estimated_cost", 1000), estimate_tokens(str(task.get("prompt", ""))))

    budget = BudgetManager(total_tokens=args.budget)
    planner = Planner(budget=budget)
    selected_tasks = planner.run(tasks)

    solver = TaskSolver(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY"),
    )

    for task in selected_tasks:
        result = solver.solve(task, task["model"])
        out_path = Path(args.out_dir) / str(task.get("task_id"))
        solver.save_answer(task, result["answer"], out_path)
        budget.consume(output_tokens=estimate_tokens(result["answer"]), input_tokens=task.get("estimated_cost", 0))

    metrics = budget.summary()
    print(json.dumps({"selected_tasks": len(selected_tasks), "metrics": metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
