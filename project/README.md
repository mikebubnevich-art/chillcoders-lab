# Budget-aware harness (openJiuwen) — priority 0 scaffold

## Run
```
docker build -t hack:latest .
docker run --rm -e LLM_BASE_URL -e LLM_API_KEY -e OUT=/tmp/out -e BUDGET=100000 \
  -v "$TASKS:/data/tasks:rw" --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --workdir /app --entrypoint python hack:latest /app/run_agent.py --tasks-dir /data/tasks
```
Env: `OUT`, `TASKS`, `LLM_BASE_URL`, `LLM_API_KEY`; optional `BUDGET` (default 100000), `MODEL`
(default `qwen3.6-35b-a3b-noreason`), `WORKERS` (<=5), `TASK_CAP` (1500), `DEADLINE_S` (3300).

## Token accounting
cost = output + reasoning + input/10, provider cache ignored (`harness/llm_client.py::cost_of`).
Every call reserves its worst case (`input_est/10 + max_tokens`) in a global locked `Budget`
before the request and commits real usage after, so 5 parallel sessions cannot overshoot.
A 3% safety margin covers rounding differences with the proxy. Per-task cap: `TASK_CAP`.

## Flow
**Phase 1 - triage, 0 tokens, all tasks, parallel** (`harness/triage.py`):
1. Template solvers (`solvers/`): `math_solver` (chain of split/double/ship/receive steps), `data_solver`
   (csv filters + sum/count/avg/min/max/revenue), `search_solver` (kb/*.md graph: link fields like Employer/Mentor,
   terminal fields like Born/Headquarters), `code_solver` (see below). A solver returns an answer, `None`
   (pattern unknown -> LLM) or `UNSOLVABLE`. When unsure a solver returns `None`; it never guesses.
2. Feasibility heuristics (`harness/feasibility.py`): file/dir referenced in the text but absent, internet/URL
   required, empty instruction -> task skipped for free (no answer written).

**Phase 2 - LLM, only for what phase 1 could not solve** (`harness/planner.py`, `harness/agent.py`):
tasks ordered by `P(success) * reward / E[tokens]` (a-priori model now, to be calibrated from trajectories),
greedy fill of 85% of the budget first. One call -> python script -> run locally -> at most one retry.

Results: `<task>/answer.txt` and `OUT/solutions.json`, written after EVERY solved task.

### code_solver (0-token bug fixing)
Enumerates single AST mutations (comparison/arith/bool flips, sorted<->reversed, min<->max, constants +-1,
slice bounds, len +-1) of the functions touched by failing tests, greedily keeps the mutation that removes the
most failures. For functions with a known reference (`REF`: reverse_words, total, maximum, minimum, average,
first_n, count_even, count_odd) it also requires behavioural equivalence on random inputs (ints AND floats) and
repairs functions that disagree with the reference even if the visible tests are green (hidden tests may be
stricter). Tests are never edited. Unknown functions are accepted by tests only.

## Non-obvious
- `expected`/`check` from task.json are never read by the agent (local tests only).
- No framework prompt is added: plain HTTP client, so the empty-call cost is tiny.
  openJiuwen is installed and pinned; wiring its model client in is a TODO (see below).
- If OUT is unwritable we fall back to /tmp/solutions.json.

## Tests (no real LLM needed)
`python tests/test_budget.py`, `test_solvers.py`, `test_triage.py`, `test_planner.py` and `python tests/e2e.py` (mock proxy, fixtures from the sample tasks;
`search_synth` and `code_synth` are synthetic). Synthetic infeasible tasks: `tools/gen_infeasible.py`. Real-model cost: `python tools/measure_overhead.py`.
