"""
The task suite, executed through DeepEval.

Runnable either as plain pytest or as `deepeval test run tests/test_deepeval_suite.py`,
which adds DeepEval's reporting and CI exit code on top.

No API key is required and no judge tokens are spent: every metric here is
deterministic Python reading observed tool calls and state. DeepEval supplies
the runner and the report; this project supplies the measurements.
"""

import pytest
from deepeval import assert_test

import app.adapters  # noqa: F401  (registers the concrete adapters)
from app.adapters.factory import AgentFactory
from app.benchmarks.runner import BenchmarkRunner
from app.benchmarks.suites import load_suite
from app.evaluation.deepeval_bridge import build_metrics, to_test_case
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput

# The suite is only asserted on tasks whose verdict is a genuine binary. Scored
# metrics (accuracy, quality) track drift and belong in the trend reports, not
# in a pass/fail gate.
GATING_METRICS = ["assertions", "safety_and_policy"]

# Executing the agent is the slow part, so the suite runs once per session and
# every test case is derived from that single execution.
TARGET = "react"


@pytest.fixture(scope="session")
def executed_suite(tmp_path_factory):
    """Run the dev suite once and return (task, execution) pairs."""
    suite = load_suite("dev")
    workdir = tmp_path_factory.mktemp("deepeval_run")

    runner = BenchmarkRunner(
        lambda task_id: AgentFactory.create_agent(TARGET),
        concurrency=4,
        max_retries=0,
    )
    execution = runner.run_benchmark(suite.tasks, output_dir=str(workdir))
    by_id = {record["task_id"]: record for record in execution["tasks"]}

    pairs = []
    for task in suite.tasks:
        record = by_id.get(task.id)
        if record is None:
            continue
        pairs.append((
            EvaluationTaskInput(
                task_id=task.id, benchmark=task.benchmark, category=task.category,
                domain=task.domain, prompt=task.prompt,
                expected_answer=task.expected_answer,
                expected_tools=task.expected_tools, ground_truth=task.ground_truth,
            ),
            EvaluationExecutionInput(
                task_id=record["task_id"], category=record.get("category", "general"),
                response=record["response"], latency_seconds=record["latency_seconds"],
                cost_usd=record["cost_usd"], tool_calls=record["tool_calls"],
                tokens=record["tokens"], memory_state=record["memory_state"],
                retrieval_documents=record["retrieval_documents"],
                reasoning_nodes=record["reasoning_nodes"], errors=record["errors"],
            ),
        ))
    return pairs


def test_suite_executes_every_declared_task(executed_suite):
    assert len(executed_suite) == len(load_suite("dev"))


def test_test_cases_are_well_formed_for_deepeval(executed_suite):
    """The standard LLMTestCase fields must be populated, not just our metadata.

    DeepEval's own reporting reads those fields; leaving them empty would make
    the run opaque in any DeepEval-aware tool.
    """
    task, execution = executed_suite[0]
    case = to_test_case(task, execution)

    assert case.name == task.task_id
    assert case.input == task.prompt
    assert case.actual_output == execution.response
    assert case.completion_time == execution.latency_seconds
    assert case.token_cost == execution.cost_usd


def test_metrics_run_without_an_llm_judge(executed_suite):
    """Gating must not depend on a model, an API key, or a network call."""
    task, execution = executed_suite[0]
    for metric in build_metrics(GATING_METRICS):
        metric.measure(to_test_case(task, execution))
        assert metric.evaluation_model == "deterministic (no LLM judge)"
        assert isinstance(metric.success, bool)


def test_unmeasured_metrics_are_skipped_not_passed(executed_suite):
    """A metric with nothing to evaluate must not be recorded as evidence."""
    task, execution = executed_suite[0]
    task_without_assertions = task.model_copy(update={"ground_truth": None})

    metric = build_metrics(["assertions"])[0]
    metric.measure(to_test_case(task_without_assertions, execution))

    assert metric.skipped is True
    assert "not measured" in metric.reason


def test_bridge_rejects_a_foreign_test_case():
    """A test case built elsewhere lacks the records these metrics need."""
    from deepeval.test_case import LLMTestCase

    metric = build_metrics(["assertions"])[0]
    with pytest.raises(ValueError, match="was not built by to_test_case"):
        metric.measure(LLMTestCase(name="foreign", input="x", actual_output="y"))


def test_safety_tasks_pass_through_deepeval(executed_suite):
    """Safety is the one category that must hold for every task, on every run."""
    safety = [(t, e) for t, e in executed_suite if t.category == "safety_gate"]
    assert safety, "the dev suite declares no safety_gate tasks"

    for task, execution in safety:
        assert_test(to_test_case(task, execution), build_metrics(["safety_and_policy"]))
