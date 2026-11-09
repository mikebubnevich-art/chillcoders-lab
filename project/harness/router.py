"""Unified routing decisions for task execution.

This layer is the single source of truth for deciding whether a task should be
solved by local deterministic logic, skipped as infeasible, or sent to the LLM
path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from solvers import UNSOLVABLE, try_solve

from . import feasibility


@dataclass
class RouteDecision:
    status: str
    answer: str | None = None
    p_feasible: float = 1.0
    reasons: list[str] = field(default_factory=list)
    route: str = ""
    reason_code: str = ""


def route_task(task: dict, use_solvers: bool = True) -> RouteDecision:
    """Return a single decision for how this task should be handled.

    States:
      - solved: local deterministic solver produced a validated answer
      - infeasible: task is not feasible and should be skipped
      - llm: route to LLM path
    """
    ans = try_solve(task, dry_run=not use_solvers)
    if ans == UNSOLVABLE:
        # Two different conventions exist: a missing/unanswerable task is
        # skipped (no answer.txt), while a task that explicitly asks to write
        # UNSOLVABLE when the data is insufficient must get that literal answer.
        if re.search(r"write\s+(?:exactly\s+|only\s+)?(?:the\s+word\s+)?UNSOLVABLE\b",
                     task.get("instruction", ""), re.I):
            return RouteDecision(
                status="solved",
                answer="UNSOLVABLE",
                reasons=["solver:unsolvable-answer"],
                route="solver",
                reason_code="unsolvable_answer",
            )
        return RouteDecision(
            status="infeasible",
            reasons=["solver:unsolvable"],
            route="skip",
            reason_code="unsolvable",
        )

    if use_solvers and ans:
        return RouteDecision(
            status="solved",
            answer=ans,
            reasons=[],
            route="solver",
            reason_code="solver",
        )

    fz = feasibility.check(task)
    if fz.hard:
        return RouteDecision(
            status="infeasible",
            p_feasible=0.0,
            reasons=fz.reasons,
            route="skip",
            reason_code="hard_feasibility",
        )

    return RouteDecision(
        status="llm",
        p_feasible=fz.p,
        reasons=fz.reasons,
        route="llm",
        reason_code="llm",
    )
