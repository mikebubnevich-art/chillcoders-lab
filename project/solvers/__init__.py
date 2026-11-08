"""Template solvers (0 LLM tokens).

try_solve(task) -> str | None | UNSOLVABLE
  str         confident answer (written to answer.txt)
  None        pattern not recognized -> agent falls back to ONE LLM call
  UNSOLVABLE  provably infeasible from local files (missing file / entity) ->
              agent spends 0 tokens and writes nothing

Rule: a solver must return None whenever it is not sure. A wrong answer costs the
same as no answer but also blocks the cheaper-than-LLM fallback.
"""
UNSOLVABLE = "__UNSOLVABLE__"

from . import code_solver, data_solver, math_solver, search_solver  # noqa: E402

_BY_TYPE = {"math": [math_solver], "data_analysis": [data_solver], "search": [search_solver],
            "coding": [code_solver], "code": [code_solver]}


def try_solve(task: dict):
    mods = _BY_TYPE.get(task.get("type"), [math_solver, data_solver, search_solver, code_solver])
    for m in mods:
        try:
            r = m.solve(task["instruction"], task["dir"])
        except Exception:  # a solver bug must never kill the task: fall back to LLM
            r = None
        if r is not None:
            return r
    return None
