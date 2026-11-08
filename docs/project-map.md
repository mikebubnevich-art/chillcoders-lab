# Project map

## External and exploratory projects

- `agent-core/` — core agent runtime and orchestration logic.
- `agent-security/` — security and validation-oriented components.
- `agent-studio/` — UI, workflow builder, experiment tooling.
- `agent-memory/` — persistent memory experiments.
- `jiuwenswarm/` — runtime helpers and warm start utilities.

## Benchmark and task materials

- `task-dir/` — task dataset used for benchmark runs.
- `benchmarks/` — structured place for benchmark scripts and result artifacts.

## Prototype and scratch files

- `model.py` — quick prototype for direct OpenAI interaction.
- `prototypes/` — experiments which should not be mixed with the final harness.

## Competition harness

- `project/` — main implementation of the competition agent.
  - `project/apps` — entry points.
  - `project/core` — budget, planner, routing.
  - `project/agents` — solver and model wrappers.
  - `project/tasks` — task loading logic.
  - `project/utils` — utility code.
  - `project/tests` — automated tests.

## Rule for future work

Keep showcase repositories under `external/`, competition data under `benchmarks/`, experiments under `prototypes/`, and production harness logic under `project/`.
