# Benchmark harness architecture

This project is organized for a token-budget-driven benchmark agent.

## Directory map

- `project/apps` — CLI entry points and startup scripts.
- `project/core` — planning, routing, and budget logic.
- `project/agents` — solver and LLM runtime integration.
- `project/tasks` — task loading and metadata parsing.
- `project/utils` — helper functions, token estimates, small utilities.
- `project/config` — constants and model registry.
- `project/tests` — regression and smoke tests.

## Typical pipeline

1. Load task directories from `TASKS`.
2. Score tasks by reward and estimated cost.
3. Select a task using a budget-aware planner.
4. Choose a suitable model via the router.
5. Solve via the LLM client.
6. Save `answer.txt` in the task directory.
7. Update budget and continue until the budget is exhausted.

## Existing workspace items

- `agent-core` — framework core, agent runtime, model abstractions.
- `agent-security` — security layer, risk and policy checks.
- `agent-studio` — UI / orchestration / experiments and tooling.
- `jiuwenswarm` — low-level runtime or environment helpers.
- `task-dir` — benchmark tasks and data set.
- `model.py` — local prototype / scratch model.
- `agent-memory` — memory storage experiments.

This project intentionally keeps external and exploratory repositories separate from the actual competition harness.
