# IMPLEMENTATION_ROADMAP

## 1. Цель архитектуры

Итоговая архитектура должна быть не “агентный монолит”, а экономичная соревновательная система с чётким разделением ответственности:

1. Discovery of tasks
2. Feature extraction
3. Classification
4. Feasibility detection
5. Cost estimation
6. Budget scheduler
7. Deterministic solvers
8. LLM gateway
9. Code solver
10. Validation
11. Global token budget manager
12. Trajectory logger

Основная идея:
- всё, что можно решить локально — решать локально;
- LLM использовать только там, где это действительно окупается;
- политику выбора маршрута сделать единой и прозрачной.

---

## 2. Целевой компонентный дизайн

### 2.1. Task Discovery

Статус: существует.

Где: `project/harness/tasks.py`, `project/run_agent.py`

Что нужно:
- нормализовать загрузку задач ещё жёстче;
- гарантировать одинаковый контракт для любых форматов layout.

Взаимодействие:
- получает список задач;
- передаёт в `triage`, `planner`, `storage`.

Тесты:
- task layout positive/negative
- missing task.json
- both layouts (`workspace/` and flat)

Решение: KEEP + IMPROVE

---

### 2.2. Feature Extraction

Статус: существует; features (тип, файлы, размеры, длина инструкции, reward,
тесты) используются планировщиком вместе с эмпирикой по типам задач.

Где: `project/harness/planner.py`

Что нужно:
- держать feature extraction отдельным блоком, а не “подмешивать” в planner;
- накапливать реальную статистику по типам задач (`p90` стоимости, success rate);
- добавлять features for: task type, file counts, file sizes, instruction length, reward, test availability.

Взаимодействие:
- `planner` -> `cost estimator`
- `triage` -> `classification`

Тесты:
- deterministic outputs for same task
- stable feature extraction across workspace layouts

Решение: IMPROVE

---

### 2.3. Task Classification

Статус: существует; единый `harness/router.py` и явные типы
`math`, `data_analysis`, `search`, `coding/code`, `shell`, `logic`.

Где: `project/harness/tasks.py`, `project/harness/agent.py`

Что нужно:
- сделать единый router classification policy;
- явно различать `math`, `data_analysis`, `search`, `coding`, `unsupported`, `unsolvable`.

Взаимодействие:
- `triage` -> route selection
- `LLM gateway` -> per-type choices

Тесты:
- classification of synthetic tasks
- classification of edge-case task descriptions

Решение: IMPROVE + REFACTOR

---

### 2.4. Feasibility Detection

Статус: существует.

Где: `project/harness/feasibility.py`, `project/harness/triage.py`

Что нужно:
- расширить heuristics на реальные failure patterns;
- чётко отделять `UNSUPPORTED` от `UNSOLVABLE`.

Взаимодействие:
- `triage` -> `planner`
- `planner` -> skip tasks with p_feasible = 0

Тесты:
- missing files
- impossible search chain
- missing CSV columns
- contradictory instructions
- absence of tests for code

Решение: KEEP + IMPROVE

---

### 2.5. Cost Estimation

Статус: существует; внедрена эмпирическая калибровка из task-level JSONL.

Где: `project/harness/planner.py`, `project/run_agent.py`, `project/harness/llm_client.py`

Что сделано:
- `LLMClient.log_task` пишет одну строку на задачу: `{type,cost,success,src,...}`;
- `load_trajectory_stats` агрегирует `avg_cost`, `p90_cost`, `success_rate`, `count`;
- `est_cost` берёт `max(prior, blend)` с весом по числу наблюдений и winsorize
  эмпирики на 3x prior: данные могут поднять оценку, но не занизить её;
- `run_agent` передаёт именно эмпирические stats в `planner.rank` (раньше
  имя `stats` перетиралось счётчиками).

Взаимодействие:
- `planner` -> `budget scheduler`
- `llm_client` -> actual cost is committed after request

Тесты:
- estimation for each type
- stress on budget edge cases
- different models produce different caps

Решение: IMPROVE + REFACTOR

---

### 2.6. Budget Scheduler

Статус: существует.

Где: `project/run_agent.py`, `project/harness/budget.py`, `project/harness/planner.py`

Что нужно:
- сделать единый scheduler, который учитывает:
  - remaining budget
  - reserve
  - priority
  - task type
  - model choice
  - actual cost
  - deadline

Взаимодействие:
- `planner` -> selected tasks
- `llm_client` -> reserved / committed

Тесты:
- over-budget protection
- parallel tasks with 5 worker limit
- no double reservation

Решение: KEEP + IMPROVE

---

### 2.7. Deterministic Solvers

Статус: существует и это один из ключевых преимуществ проекта. На
`generated_tasks/` детерминированное покрытие 12/12 (включая `shell`, `logic` и
`UNSOLVABLE`), на полном `task-dir` — 20/20, оба прогона за 0 токенов.

Где: `project/solvers/`, `project/harness/dsl.py`

Что нужно:
- поддерживать их как primary path;
- расширять набор распознаваемых шаблонов;
- развивать coverage и реальную библиотеку формулировок;
- добавлять детальный status: `SOLVED`, `UNSUPPORTED`, `UNSOLVABLE`, `ERROR`.

Взаимодействие:
- `triage` -> deterministic solve first
- `LLM gateway` -> only if unsupported or failed

Тесты:
- math cases with variants of wording
- csv filters / groups / aggregations
- local KB traversal and stop conditions

Решение: KEEP + IMPROVE

---

### 2.8. LLM Gateway

Статус: существует.

Где: `project/harness/llm_client.py`, `project/harness/oj_transport.py`

Что нужно:
- держать один gateway API;
- все запросы проходят через этот слой;
- усилить logging, retries, fallback rules.

Взаимодействие:
- `agent.solve_llm`
- `budget`
- `traj logger`

Тесты:
- timeout
- 429 handling
- fallback to urllib
- cost accounting with usage metadata

Решение: KEEP + IMPROVE

---

### 2.9. Code Solver

Статус: exists, major strength.

Где: `project/harness/agent.py`, `project/harness/patch.py`, `project/harness/runner.py`

Что нужно:
- keep patch-first strategy;
- continue script fallback only when patch cannot express the fix;
- ensure robust verification before acceptance.

Взаимодействие:
- `runner` for tests
- `storage` for `DONE`/answer.txt
- `budget` for actual cost

Тесты:
- patch passes verifier
- bad patch falls back to script
- code with hidden tests path remains protected

Решение: KEEP + IMPROVE

---

### 2.10. Validation

Статус: существует хорошо.

Где: `project/harness/agent.py`, `project/tests/`, `project/harness/runner.py`

Что нужно:
- все решения должны подтверждаться локально перед возвратом `DONE`;
- validation must be a hard gate, not an afterthought.

Взаимодействие:
- `code solver` -> validation
- task result -> storage

Тесты:
- successful fix passes original tests
- false DONE without fix is rejected
- sandbox restore before verification

Решение: KEEP

---

### 2.11. Global Token Budget Manager

Статус: существует strongly.

Где: `project/harness/budget.py`, `project/harness/llm_client.py`

Что нужно:
- keep as source of truth;
- ensure all reservations are centralized and visible;
- log remaining budget per wave and per task.

Взаимодействие:
- all LLM calls
- planner
- runner (task costs and result acceptance)

Тесты:
- exact budget limits
- concurrency protection
- no overshoot under parallel execution

Решение: KEEP

---

### 2.12. Trajectory Logger

Статус: существует; per-call rows + task-level rows, пригодные для
калибровки планировщика.

Где: `project/harness/llm_client.py`, `project/run_agent.py`

Что нужно:
- держать два уровня: запрос (`tag/model/usage/cost/transport/elapsed`) и задача
  (`type/cost/success/src/model/reward/sec`);
- агрегировать по `type/model/src` в `tools/model_matrix.py`;
- не смешивать task-level строки с per-call при расчёте p50/p95.

Взаимодействие:
- `llm_client` -> logger
- `planner` <- past performance stats

Тесты:
- valid JSONL writes
- retention after multiple runs
- parseable aggregated metrics

Решение: IMPROVE

---

## 3. Приоритетная последовательность работ

### Порядок 1: policy consolidation

Проблема: LLM-routing logic is spread across too many files.

Что делать:
- единственный router; один вход; одно решение по типу задачи;
- `triage` -> route result -> execution policy.

Зависимости:
- must happen before advanced tuning because otherwise experiments are noisy.

---

### Порядок 2: empirical cost calibration

Проблема: planner heuristics are still rough.

Что делать:
- collect real telemetry by task type and model;
- recalibrate `P(success)`, `expected_cost`, `reward efficiency`;
- introduce a simple regression or ranking on historical data.

Зависимости:
- depends on stable logger and router.

---

### Порядок 3: harden openJiuwen + budget behaviour

Проблема: proxy or adapter instability is a tournament risk.

Что делать:
- live proxy test matrix;
- 429, timeout and 5xx handling;
- verify fallback behaviour with real provider behaviour.

Зависимости:
- requires stable LLM layer.

---

### Порядок 4: expand deterministic coverage

Проблема: many tasks still go to LLM unnecessarily.

Что делать:
- expand math/data/search solver coverage;
- improve `UNSUPPORTED` detection;
- reject tasks as `UNSOLVABLE` earlier.

Зависимости:
- needed before final run to maximize reward per token.

---

### Порядок 5: final benchmark + model lock

Проблема: need final stable model configuration.

Что делать:
- full synthetic benchmark
- real task sample benchmark
- final choose of `MODEL`, `MODEL_CODING`, `CODING_STRATEGY`
- freeze config for final competition run.

Зависимости:
- after stability and logging checks.

---

## 4. Рекомендованная итоговая конфигурация (текущая рабочая гипотеза)

На основе текущих документов и кода гипотеза следующая:

- base model: `qwen3.8-27b-noreason`
- coding model: `qwen3.8-27b`
- coding strategy: `fallback`
- keep deterministic solvers as first class;
- LLM only for unsupported / failed / code tasks;
- budget reserve: conservative, important for 100 tasks;
- use DSL before generic script;
- use patch first before general script for code.

Это не финальный вывод, а рабочая стратегия, которую нужно подтвердить во время benchmark и real-run validation.

---

## 5. Критерии готовности к финальному запуску

Проект считается готовым к финальному турнирному запуску, когда:

- все критические тесты проходят;
- budget protection is proven on parallel runs;
- openJiuwen / raw fallback strategy is stable;
- deterministic solver coverage is high enough;
- planner chooses tasks in a stable, explainable way;
- no task category silently falls through without explicit route.

---

## 6. Итог по roadmap

Текущий проект уже достаточно зрелый, чтобы не начинать заново. Он нуждается не в “масштабном рефакторинге”, а в:
- consolidation of policy,
- calibration of cost model,
- validation on real scenarios,
- final freeze of model strategy.

Это правильный порядок действий для достижения максимального reward при бюджете 100k.
