# Budget-aware harness (openJiuwen) — priority 0 scaffold

## Run
```
docker build -t hack:latest .
docker run --rm -e LLM_BASE_URL -e LLM_API_KEY -e OUT=/tmp/out -e BUDGET=100000 \
  -v "$TASKS:/data/tasks:rw" --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --workdir /app --entrypoint python hack:latest /app/run_agent.py --tasks-dir /data/tasks
```
Env: `OUT`, `TASKS`, `LLM_BASE_URL`, `LLM_API_KEY`; optional `BUDGET` (default 100000), `MODEL`
(default `qwen3.8-27b-noreason`), `MODEL_CODING` (default `qwen3.8-27b`; empty = same as `MODEL`),
`CODING_STRATEGY` (`fallback` by default: base model first, strong model only after a failed coding
attempt; `always`: every coding task goes straight to `MODEL_CODING`),
`WORKERS` (<=5), `TASK_CAP` (1500 base), `DEADLINE_S` (3300),
`SANDBOX_MAX_MB` (256), `SANDBOX_MEM_MB` (4096), `SANDBOX_FSIZE_MB` (32), `DISABLE_DSL=1`,
`DISABLE_PATCH=1`, `DISABLE_SOLVERS=1`, `USE_OJIWEN=0` (force the stdlib transport).

Recommended competition config is **hybrid**: `qwen3.8-27b-noreason` for everything, with
`qwen3.8-27b` as a surgical retry for coding tasks that the base model failed. The rules allow switching
models on any call. `CODING_STRATEGY=always` sends every coding task to the strong model instead;
`MODEL_CODING=` (empty) disables the switch.

## Task layout
`--tasks-dir` accepts both layouts used by the organizers:
`<task>/workspace/{sales.csv,kb/,mylib.py,...}` (the real samples) and flat `<task>/{...}`.
`harness/tasks.py` picks `workspace/` as the working directory when it exists, so solvers, DSL, patch mode,
sandbox and tests all run where the instruction says "current directory". `answer.txt` is written both to the
working dir and to `<task>/` (the rules say "in each task directory"), so either checker location sees it.
`task.json` is always read from `<task>/`; `expected`/`check` are never read by the agent.

The 8 real samples in `task-dir/` are solved by phase 1 for **20/20 reward and 0 tokens** (checked manually;
`tests/test_layout.py` covers both layouts on synthetic copies, and e2e runs coding through patch mode in
`workspace/`).

## Token accounting
cost = output + reasoning + input/10, provider cache ignored (`harness/llm_client.py::cost_of`).
Every call reserves its worst case (`input_est/10 + max_tokens`) in a global locked `Budget`
before the request and commits real usage after, so 5 parallel sessions cannot overshoot.
A 3% safety margin covers rounding differences with the proxy. Per-task cap is
`max(TASK_CAP, 2 * max_output + 600)` for the task type, so the first attempt plus one
retry always fit (coding: 2400). LLM requests are deadline-aware: the HTTP timeout shrinks
to the remaining `DEADLINE_S`, and no new request starts in the last 5 seconds.

## openJiuwen transport
`harness/oj_transport.py` uses openJiuwen 0.1.19 `OpenAIModelClient` as the preferred HTTP transport, so the
harness really runs on top of openJiuwen while budget, deadlines, retries and prompts stay in our code. It
configures openJiuwen logging under `/tmp` (the container is read-only), builds a fresh async client per call
(safe across the 5 worker threads/event loops), and maps `usage_metadata` to the OpenAI usage dict. If
openJiuwen is absent or a request through it fails, `LLMClient` falls back to raw `urllib` within the same
budget reservation and disables the adapter for the rest of the run. `USE_OJIWEN=0` forces the stdlib path for
A/B measurement; `tests/e2e.py::test_openjiuwen_transport_is_used` runs only where the full runtime is installed.

## Flow
**Phase 1 - triage, 0 tokens, all tasks, parallel** (`harness/triage.py`):
1. Template solvers (`solvers/`): `math_solver` (chain of split/double/ship/receive steps, lots, fractions with
   rounding, bundles), `data_solver` (csv filters + sum/count/avg/min/max/revenue/percent/sum_product),
   `search_solver` (kb/*.md graph: link fields like Employer/Mentor, terminal fields like Born/Headquarters,
   plus facts stated in prose), `code_solver` (see below), `shell_solver` (largest/smallest even-odd, sum,
   count over one-integer-per-line files) and `logic_solver` (provable inconsistency -> `UNSOLVABLE`). A solver
   returns an answer, `None` (pattern unknown -> LLM) or `UNSOLVABLE`. When unsure a solver returns `None`; it
   never guesses.
2. Feasibility heuristics (`harness/feasibility.py`): file/dir referenced in the text but absent, internet/URL
   required, empty instruction -> task skipped for free (no answer written).

**Phase 2 - LLM, only for what phase 1 could not solve** (`harness/planner.py`, `harness/agent.py`,
`harness/dsl.py`):
tasks ordered by `P(success) * reward / E[tokens]`, greedy fill of 85% of the budget. **Two waves**: wave 1
runs only the planner-selected tasks; after it finishes, wave 2 admits filler tasks greedily against the
**actual** remaining budget.

The planner is calibrated from past runs but stays conservative: `LLMClient.log_task` writes one row per task
(`{type,cost,success,src,...}`), `load_trajectory_stats` aggregates `avg_cost`, `p90_cost`, `success_rate` and
`count` per type, and `est_cost` returns `max(prior, blend)`. Empirical data may raise an estimate but never
lower it below the a-priori envelope; the empirical value is winsorized at `3*prior` and weighted by
`min(0.5, count/20)`, so a single lucky cheap run cannot under-reserve the budget.

For math/data/search the first call asks the model to **compile** the task into a compact JSON plan
(math ops, CSV filters+aggregate, KB hops). `harness/dsl.py` parses, validates and executes it locally:
`Fraction` arithmetic for math, `csv` filters/aggregates for data, the KB graph for search. Data aggregates:
`sum`, `count`, `count_distinct`, `avg`, `median`, `min`, `max`, `group_top`, `percent` (share of filtered
rows) and `sum_product` (e.g. revenue = `sum(units*price)`). A plan that is
invalid, ambiguous or not exactly computable is rejected with a short error; the second call then asks for a
full python3 script (the generic fallback). Thus at most 2 calls per task, and the expensive script is only
used when the plan does not fit. `DISABLE_DSL=1` forces the script-only mode for A/B measurement.

For coding tasks the first call asks for **exact text replacements** (`harness/patch.py`): the model quotes the
buggy fragment and its replacement, including several hunks/files. Local code verifies every `old` occurs
exactly once, rejects test-file/hidden/escaping paths, applies the edits to the sandbox, writes `DONE`, and runs
the original tests. A rejected plan or failing tests fall back to the full-script mode with the failure as
feedback. `DISABLE_PATCH=1` disables this mode.

A script runs in a disposable copy of the task dir (`harness/runner.py`): rlimits for CPU, address
space, file size and (on Linux) process count, a wall-clock timeout with process-group kill, and a clean
environment. Only `answer.txt` is copied back; for coding tasks, changed non-test `.py` modules are copied
back too. Very large task dirs (> `SANDBOX_MAX_MB`) fall back to in-place execution with the same limits and
in-memory test snapshots.

Coding tasks get a verifier: the answer is accepted only when `answer.txt` contains the marker requested by the
instruction (`PASS`, `DONE`, …) or the legacy `DONE`, **and** the test command from the instruction
(e.g. `python3 test_mylib.py`, or discovered `test_*.py`) exits 0. Test files are restored from a snapshot
before verification, so editing/deleting tests cannot make a task look solved. The stored answer is normalized
to the requested marker (`harness/storage.py::normalize_coding_answer`).

Results: `answer.txt` in the working dir and in `<task>/` (when they differ), plus `OUT/solutions.json`,
written after EVERY solved task.

### code_solver (0-token bug fixing)
Enumerates single AST mutations (comparison/arith/bool flips, sorted<->reversed, min<->max, constants +-1,
slice bounds, len +-1), plus two structural repairs: wrapping a `min`/`max` return in an ignored parameter
(clamp-like bug) and rotating `a + b / 2` into `(a + b) / 2` (even-median). It greedily keeps the mutation
that removes the most failures. For functions with a known reference (`REF`: reverse_words, total, maximum,
minimum, average, first_n, count_even, count_odd) it also requires behavioural equivalence on random inputs
(ints AND floats) and repairs functions that disagree with the reference even if the visible tests are green
(hidden tests may be stricter). Tests are never edited. Unknown functions are accepted by tests only. The
returned answer is the marker requested by the instruction.

## Non-obvious
- `expected`/`check` from task.json are never read by the agent (local tests only).
- Two infeasible conventions coexist. If the instruction explicitly says to write `UNSOLVABLE` when the value
  cannot be determined, the literal answer `UNSOLVABLE` is written; otherwise a provably infeasible task is
  skipped with no `answer.txt`. The literal-answer branch is gated on the `write … UNSOLVABLE` wording, not on
  any mention of the word.
- A coding answer is never trusted from `answer.txt` alone: the original tests must pass locally.
- A missing file referenced as an OUTPUT (`write the result to results.csv`) is not treated as infeasible;
  a missing INPUT file still is.
- No framework prompt is added: plain HTTP client, so the empty-call cost is tiny. The openJiuwen transport
  is below it (see «openJiuwen transport») and keeps prompts/cost accounting ours.
- If OUT is unwritable we fall back to /tmp/solutions.json.

## Tests (no real LLM needed)
`python -m unittest discover -s tests -p 'test_*.py'` (141 tests) and `python tests/e2e.py` (16 tests, mock
proxy, fixtures from the sample tasks; `search_synth` and `code_synth` are synthetic). The e2e suite covers
two-wave scheduling, DONE-without-fix rejection, test tampering, a successful LLM fix, sandbox isolation,
timeouts, DSL plans, the DSL->script fallback, coding patch plans, both task layouts (`workspace/` and flat),
and planner calibration from a trajectory file. Synthetic infeasible tasks: `tools/gen_infeasible.py`.
Real-model cost: `python tools/measure_overhead.py`.

## Committed deterministic benchmark
`generated_tasks/` (12 tasks: math, data, search, shell, coding, logic, two unsatisfiable) is the regression
corpus for phase 1. Current result: **12/12 correct, 0 wrong, 0 tokens, reward 35**; the full `task-dir` set is
**20/20, 0 tokens, reward 55**. The set is a real workspace input, so tests copy tasks to a temp dir before
calling `route_task`/`try_solve` with writes enabled. `tests/test_solvers.py::GeneratedBenchmark` pins the
expected answers (including the literal `UNSOLVABLE` for the two infeasible ones).

## Synthetic 100-task benchmark
```
python tools/gen_synthetic.py /tmp/synth100     # 25 math, 25 data, 20 search, 15 coding, 15 infeasible
python tools/bench_solvers.py /tmp/synth100 --min-coverage 0.7
```
`tools/bench_solvers.py` runs phase-1 triage and reports zero-token coverage, **wrong answers**
(solved but != `expected`), false skips and infeasible detection. Ground truth is computed by the
generator independently, so a solver bug cannot hide behind shared code. Current result on seed 7:
coverage 100% (85/85 feasible), wrong-answer rate 0%, infeasible caught 100% (15/15), false skips 0,
~1.5s for all 100 tasks. The suite includes this run as a regression test (seed 11).

## Model matrix (all four competition models)
Credentials are never committed. Put the proxy into `project/.env.local` (gitignored):
```
LLM_BASE_URL=https://<proxy-host>/v1
LLM_API_KEY=<key>
```
`tools/model_matrix.py` loads that file automatically; `LLM_BASE_URL`/`LLM_API_KEY` from the shell win.
Then:
```
python tools/model_matrix.py --tasks /tmp/synth100 --limit 8 --budget 30000   # pilot
python tools/model_matrix.py --tasks /tmp/synth100 --budget 100000           # full matrix
python tools/model_matrix.py --tasks /tmp/synth100 --budget 100000 \
  --models qwen3.8-27b-noreason --coding-model qwen3.8-27b                  # hybrid
```
Runs are sequential (each model gets <=5 workers, so the 5-session limit holds), every model gets a fresh
copy of the tasks. Default is `--no-solvers`: solver answers are ignored to measure the model itself, but
UNSOLVABLE detection still runs so infeasible tasks are not hallucinated. Pass `--solvers` to measure the
whole harness. Correctness is checked against `task.json.expected` (dev tool only). Outputs:
`<out>/<model>/trajectory.jsonl`, `run.log`, `out/solutions.json`, plus `matrix.json` and `matrix.md`
with ok/wrong/unknown, infeasible false positives, tokens, cost per correct answer, reward per 1k tokens,
transport mix and p50/p95 latency.

Proxy limits: `--workers` is sessions, `--concurrency` is simultaneous HTTP requests (default 3). The
client lowers the limit by one on every 429 and does not disable openJiuwen for transient errors.
Reasoning (non-`-noreason`) models get `REASON_MULT` larger caps (default 4x): at the old caps 100% of
`qwen3.6-35b-a3b` calls ended at `finish_reason=length` with empty content.

Full matrix (100 tasks, 100k per model, no solvers, 2026-11-09):

| model | ok | wrong | infeasible FP | spent | cost/ok | wall |
|---|---:|---:|---:|---:|---:|---:|
| `qwen3.8-27b` | 80 | 4 | 0/15 | 55878 | 699 | 432s |
| `qwen3.8-27b-noreason` | 81 | 2 | 0/15 | 15682 | 194 | 86s |
| `qwen3.6-35b-a3b` | 79 | 3 | 0/15 | 64772 | 820 | 181s |
| `qwen3.6-35b-a3b-noreason` | 73 | 3 | 0/15 | 12061 | 165 | 54s |

After adding `percent`/`sum_product` the affected tasks were rechecked: 83 / 83 / 81 / 75 out of 85.
`qwen3.8-27b-noreason` is the base model: same coverage as the reasoning variant at ~1/3.5 the cost
and ~5x faster, and 13/15 coding vs 15/15 (reasoning coding costs 36.7k vs 9.6k).

Hybrid, `CODING_STRATEGY=fallback` (default): base `qwen3.8-27b-noreason` codes first, and only a
failed coding task is retried with `qwen3.8-27b`. Full synth100: **85/85, wrong 0, infeasible FP
0/15, 12 646 tokens, 165 s**. Breakdown: coding 15/15 at 6.6k, data 25/25 at 2.2k, math 25/25 at
3.0k, search 20/20 at 0.8k. In this run the base model solved every coding task, so the fallback did
not fire; the switch itself is covered by an e2e test. A coding-only rerun (15 tasks) was 15/15 at
5 610 tokens in 27 s. With `CODING_STRATEGY=always` (all coding
straight to the strong model) the same set is **85/85, 39 549 tokens, 190 s**, coding 33.5k. The rules
allow switching models on any call. `MODEL_CODING=` disables the switch. With solvers on, the full
synthetic set is 85/85 feasible solved, 15/15 infeasible caught, 0 tokens.
