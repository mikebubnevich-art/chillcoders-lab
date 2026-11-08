"""Global thread-safe token budget.

Cost unit = "output-equivalent tokens": output + reasoning + input / 10.
Every LLM call must `reserve` its worst case BEFORE the request and `commit`
the real usage AFTER, so 5 parallel sessions can never overshoot the total.
"""
import threading
import time


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, total: float, safety: float = 0.97):
        # safety: the proxy's rounding may differ from ours, keep a margin.
        self.total = float(total) * safety
        self.spent = 0.0
        self.reserved = 0.0
        self._lock = threading.Condition()

    def remaining(self) -> float:
        with self._lock:
            return self.total - self.spent - self.reserved

    def reserve(self, n: float, wait: float = 120.0) -> None:
        """Reserve worst-case cost. If the budget is only blocked by OTHER tasks' reservations,
        wait for them to commit (they usually cost far less than their worst case);
        raise only when the spent budget alone can no longer cover `n`."""
        deadline = time.time() + wait
        with self._lock:
            while True:
                if self.spent + n > self.total:
                    raise BudgetExceeded(f"need {n:.0f}, left {self.total - self.spent:.0f}")
                if self.spent + self.reserved + n <= self.total:
                    self.reserved += n
                    return
                left = deadline - time.time()
                if left <= 0:
                    raise BudgetExceeded("timed out waiting for reservations to clear")
                self._lock.wait(timeout=min(1.0, left))

    def commit(self, reserved: float, actual: float) -> None:
        with self._lock:
            self.reserved = max(0.0, self.reserved - reserved)
            self.spent += actual
            self._lock.notify_all()

    def snapshot(self) -> dict:
        with self._lock:
            return {"total": self.total, "spent": round(self.spent, 1),
                    "reserved": round(self.reserved, 1)}


class TaskMeter:
    """Per-task cap on top of the global budget (1.5-2x of the estimate later)."""

    def __init__(self, budget: Budget, cap: float):
        self.budget, self.cap, self.spent = budget, float(cap), 0.0

    def room(self) -> float:
        return self.cap - self.spent  # global limit is enforced (with waiting) by Budget.reserve
