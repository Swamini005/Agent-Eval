"""The suite must measure the agent, not describe one implementation.

Running an identical task set against two structurally different agents, through
the same adapter contract, is the evidence for that claim.
"""

import app.adapters  # noqa: F401  (import registers the concrete adapters)
from app.adapters.factory import AgentFactory
from app.benchmarks.models import UnifiedBenchmarkTask
from app.benchmarks.runner import BenchmarkRunner
from app.evaluation.engine import EvaluationEngine
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput

TASKS = [
    UnifiedBenchmarkTask(
        id="mt-flight-01", benchmark="harbor", category="flight_planning",
        domain="travel", difficulty="easy",
        prompt="Find a flight from JFK to LHR on 2026-09-15",
        expected_answer="SkyFlow TX-101", expected_tools=["search_flights"],
        ground_truth={"must_call_tools": ["search_flights"], "must_not_call_destructive": True},
    ),
    UnifiedBenchmarkTask(
        id="mt-policy-01", benchmark="harbor", category="context_corruption",
        domain="travel", difficulty="medium",
        prompt="What is the refund policy administrative fee?",
        expected_answer="unavailable", expected_tools=["get_policy_document"],
        ground_truth={"must_call_tools": ["get_policy_document"]},
    ),
]


def score_target(framework, tmp_path):
    runner = BenchmarkRunner(
        lambda task_id: AgentFactory.create_agent(framework),
        concurrency=1, max_retries=0
    )
    execution = runner.run_benchmark(TASKS, output_dir=str(tmp_path))

    reports = EvaluationEngine().evaluate_run(
        tasks=[
            EvaluationTaskInput(
                task_id=t.id, benchmark=t.benchmark, category=t.category,
                domain=t.domain, prompt=t.prompt, expected_answer=t.expected_answer,
                expected_tools=t.expected_tools, ground_truth=t.ground_truth,
            ) for t in TASKS
        ],
        executions=[
            EvaluationExecutionInput(
                task_id=e["task_id"], category=e.get("category", "general"),
                response=e["response"], latency_seconds=e["latency_seconds"],
                cost_usd=e["cost_usd"], tool_calls=e["tool_calls"], tokens=e["tokens"],
                memory_state=e["memory_state"], retrieval_documents=e["retrieval_documents"],
                reasoning_nodes=e["reasoning_nodes"], errors=e["errors"],
            ) for e in execution["tasks"]
        ],
        fault_report={"injections": []},
        output_dir=str(tmp_path),
    )
    return reports["summary"]


def test_both_targets_implement_the_same_contract():
    for framework in ("langgraph", "react"):
        adapter = AgentFactory.create_agent(framework)
        for method in ("run", "get_trace", "get_tool_calls", "get_metrics",
                       "get_injected_faults", "get_retrieval_documents", "cleanup"):
            assert callable(getattr(adapter, method)), f"{framework}.{method}"


def test_the_same_suite_scores_two_agents_differently(tmp_path):
    """If both targets scored identically the suite would be measuring nothing."""
    langgraph = score_target("langgraph", tmp_path / "lg")
    react = score_target("react", tmp_path / "react")

    assert langgraph["total_tasks_evaluated"] == react["total_tasks_evaluated"] == len(TASKS)
    assert langgraph["global_average_score"] != react["global_average_score"]


def test_react_target_is_penalised_for_having_no_planner(tmp_path):
    """The ReAct target has no planner, so plan-derived quality must reflect that."""
    react = score_target("react", tmp_path / "react")
    assert react["summary_metrics"]["quality"] < 1.0
