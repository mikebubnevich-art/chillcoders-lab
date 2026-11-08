from project.core.budget import BudgetManager


def test_budget_decreases() -> None:
    budget = BudgetManager(total_tokens=1000)
    assert budget.consume(output_tokens=200, input_tokens=0) is True
    assert budget.remaining_tokens == 800


def test_budget_rejects_overflow() -> None:
    budget = BudgetManager(total_tokens=100)
    assert budget.consume(output_tokens=90, input_tokens=0) is True
    assert budget.consume(output_tokens=20, input_tokens=0) is False
