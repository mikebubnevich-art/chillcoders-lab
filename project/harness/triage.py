"""Phase 1 for every task, 0 tokens: template solver -> feasibility heuristics."""
import os
from dataclasses import dataclass, field

from solvers import UNSOLVABLE, try_solve
from . import feasibility


@dataclass
class Triage:
    status: str                  # solved | infeasible | llm
    answer: str | None = None
    p_feasible: float = 1.0
    reasons: list = field(default_factory=list)


def triage(task: dict, use_solvers: bool = True) -> Triage:
    if use_solvers:
        ans = try_solve(task)
        if ans == UNSOLVABLE:
            return Triage("infeasible", reasons=["solver:unsolvable"])
        if ans:
            return Triage("solved", answer=ans)
    fz = feasibility.check(task)
    if fz.hard:
        return Triage("infeasible", p_feasible=0.0, reasons=fz.reasons)
    return Triage("llm", p_feasible=fz.p, reasons=fz.reasons)
