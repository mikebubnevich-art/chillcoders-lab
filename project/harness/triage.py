"""Phase 1 for every task, 0 tokens: template solver -> feasibility heuristics."""
from dataclasses import dataclass, field

from .router import route_task


@dataclass
class Triage:
    status: str                  # solved | infeasible | llm
    answer: str | None = None
    p_feasible: float = 1.0
    reasons: list = field(default_factory=list)


def triage(task: dict, use_solvers: bool = True) -> Triage:
    """Backward-compatible wrapper around the single routing policy."""
    decision = route_task(task, use_solvers=use_solvers)
    return Triage(
        status=decision.status,
        answer=decision.answer,
        p_feasible=decision.p_feasible,
        reasons=decision.reasons,
    )
