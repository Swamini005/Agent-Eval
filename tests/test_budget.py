"""Run budgets: stop at a declared ceiling instead of hitting the provider's.

Measured against Groq's free tier: 100,000 tokens/day, and the LangGraph agent
spends ~10,000 on one task because it re-sends a growing history each loop. Ten
tasks per day. Without a guard, a 30-task run burns the quota and the remaining
tasks fail with 429s that read like agent failures in the report.
"""

import pytest

from app.budget import BudgetExceeded, RunBudget


def test_a_budget_with_no_ceiling_never_interferes():
    budget = RunBudget()
    assert budget.enabled is False
    for _ in range(1000):
        budget.charge(10_000, 1.0)


def test_token_ceiling_stops_the_run_and_says_why():
    budget = RunBudget(max_tokens=25_000)
    budget.charge(10_000, 0.01)
    budget.charge(10_000, 0.01)

    with pytest.raises(BudgetExceeded, match="Token budget exhausted"):
        budget.charge(10_000, 0.01)


def test_the_task_that_crossed_the_line_is_still_counted():
    """Totals must include the overrunning task, not silently omit it."""
    budget = RunBudget(max_tokens=1_000)
    with pytest.raises(BudgetExceeded):
        budget.charge(1_500, 0.5)

    assert budget.tokens == 1_500
    assert budget.tasks_charged == 1


def test_cost_ceiling_is_enforced_independently():
    budget = RunBudget(max_cost_usd=0.05)
    with pytest.raises(BudgetExceeded, match="Cost budget exhausted"):
        budget.charge(10, 0.06)


def test_remaining_tasks_is_estimated_from_observed_usage():
    budget = RunBudget(max_tokens=100_000)
    budget.charge(10_000, 0.0)
    # 90,000 left at 10,000 per task.
    assert budget.remaining_tasks() == 9


def test_summary_reports_per_task_burn_rate():
    budget = RunBudget(max_tokens=100_000)
    budget.charge(12_000, 0.01)
    budget.charge(8_000, 0.01)

    summary = budget.summary()
    assert summary["tokens_used"] == 20_000
    assert summary["tokens_per_task"] == 10_000
    assert summary["estimated_tasks_remaining"] == 8


def test_runner_marks_a_budget_stopped_run_as_incomplete(tmp_path):
    """A partial run must not read as a complete measurement of the agent."""
    from unittest.mock import MagicMock
    from app.benchmarks.models import UnifiedBenchmarkTask
    from app.benchmarks.runner import BenchmarkRunner

    tasks = [
        UnifiedBenchmarkTask(id=f"t{i}", benchmark="b", category="c", domain="travel",
                             difficulty="easy", prompt="p" * 400)
        for i in range(4)
    ]

    def factory(task_id):
        adapter = MagicMock()
        adapter.run.return_value = {"response": "x" * 400, "plan": [], "intent": {}}
        adapter.get_trace.return_value = []
        adapter.get_tool_calls.return_value = []
        adapter.get_execution_graph.return_value = {}
        adapter.get_metrics.return_value = {}
        adapter.get_injected_faults.return_value = []
        adapter.get_retrieval_documents.return_value = []
        return adapter

    runner = BenchmarkRunner(factory, concurrency=1, max_retries=0,
                             budget=RunBudget(max_tokens=250))
    report = runner.run_benchmark(tasks, output_dir=str(tmp_path))

    assert report["summary"]["complete"] is False
    assert "Token budget exhausted" in report["summary"]["budget_stop_reason"]
    assert report["summary"]["budget"]["tokens_used"] > 0
