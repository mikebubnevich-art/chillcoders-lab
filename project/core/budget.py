from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetManager:
    total_tokens: int
    spent_tokens: int = 0
    remaining_tokens: int | None = None

    def __post_init__(self) -> None:
        self.remaining_tokens = self.total_tokens

    def consume(self, output_tokens: int, input_tokens: int = 0) -> bool:
        cost = output_tokens + (input_tokens // 10)
        if self.remaining_tokens is None:
            raise RuntimeError("Budget is not initialized")
        if cost > self.remaining_tokens:
            return False
        self.spent_tokens += cost
        self.remaining_tokens -= cost
        return True

    def can_afford(self, estimated_tokens: int) -> bool:
        if self.remaining_tokens is None:
            return False
        return estimated_tokens <= self.remaining_tokens

    @property
    def usage_ratio(self) -> float:
        if self.total_tokens <= 0:
            return 0.0
        return self.spent_tokens / self.total_tokens

    def summary(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "spent_tokens": self.spent_tokens,
            "remaining_tokens": self.remaining_tokens,
            "usage_ratio": round(self.usage_ratio, 4),
        }
