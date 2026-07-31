import os
import json
import pytest
from app.evaluation.engine import EvaluationEngine
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput

def test_evaluation_engine_metrics_and_reports(tmp_path):
    engine = EvaluationEngine()
    
    # Assert that all standard metrics are loaded
    metric_names = [m.name for m in engine.metrics]
    assert "accuracy" in metric_names
    assert "tool_accuracy" in metric_names
    assert "performance" in metric_names
    assert "quality" in metric_names
    assert "memory_and_retrieval" in metric_names
    assert "fault_metrics" in metric_names

    # Mock Task
    task = EvaluationTaskInput(
        task_id="t1",
        benchmark="harbor",
        prompt="Search flight JFK to LHR on 2026-09-01",
        expected_answer="SkyFlow TX-101",
        expected_tools=["search_flights"]
    )
    
    # Mock Execution Telemetry
    execution = EvaluationExecutionInput(
        task_id="t1",
        response="Found SkyFlow TX-101 departing JFK.",
        latency_seconds=2.5,
        cost_usd=0.0012,
        tool_calls=[{"tool_name": "search_flights", "args": {}}],
        tokens={"prompt_tokens": 100, "completion_tokens": 50},
        memory_state=[{"role": "user", "content": "hello"}],
        retrieval_documents=[{"source": "docs", "content": "SkyFlow TX-101 departs JFK at 08:00"}],
        reasoning_nodes=[{"node_id": "step1", "status": "completed"}],
        errors=[]
    )
    
    # Mock Fault Report
    fault_report = {
        "summary": {"total_faults_injected": 1},
        "injections": [
            {
                "fault_id": "F-01",
                "severity": "warning",
                "component": "tool"
            }
        ]
    }
    
    # Run evaluation
    reports = engine.evaluate_run(
        tasks=[task],
        executions=[execution],
        fault_report=fault_report,
        output_dir=str(tmp_path)
    )
    
    # Assert return payloads
    assert len(reports["results"]) == 1
    assert reports["results"][0]["overall_score"] > 0.5
    assert reports["results"][0]["metrics"]["accuracy"] > 0.5
    assert reports["results"][0]["metrics"]["tool_accuracy"] == 1.0 # Matched search_flights
    
    # Verify outputs
    results_file = os.path.join(tmp_path, "results.json")
    benchmark_file = os.path.join(tmp_path, "benchmark_report.json")
    agent_file = os.path.join(tmp_path, "agent_report.json")
    summary_file = os.path.join(tmp_path, "evaluation_summary.json")
    
    assert os.path.exists(results_file)
    assert os.path.exists(benchmark_file)
    assert os.path.exists(agent_file)
    assert os.path.exists(summary_file)
    
    with open(summary_file, "r") as f:
        summary = json.load(f)
        assert summary["total_tasks_evaluated"] == 1
        assert "global_average_score" in summary


def test_a_task_with_no_expected_answer_is_unmeasured_not_scored():
    """`target in actual` is trivially true for an empty target.

    A task declaring no expected answer scored 0.4 for arbitrary output, and a
    full 1.0 when the agent returned nothing at all -- so the emptiest possible
    response was the highest scoring one. Nothing to compare against is not the
    same as a wrong answer, and must be excluded from the aggregate instead.
    """
    from app.evaluation.metrics.accuracy import AccuracyMetric
    from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput

    def evaluate(expected, response):
        task = EvaluationTaskInput(task_id="x", benchmark="b", category="c", difficulty="easy",
                                   domain="travel", prompt="p", expected_answer=expected)
        execution = EvaluationExecutionInput(
            task_id="x", category="c", response=response, latency_seconds=1.0, cost_usd=0.0,
            tool_calls=[], tokens={}, memory_state=[], retrieval_documents=[],
            reasoning_nodes=[], errors=[])
        return AccuracyMetric().evaluate(task, execution, {})

    for response in ("", "complete nonsense"):
        result = evaluate("", response)
        assert result.measured is False, response

    # A real reference answer is still scored normally.
    assert evaluate("450 dollars", "450 dollars").measured is True
    assert evaluate("450 dollars", "450 dollars").score == 1.0


def test_a_crashed_task_is_not_scored_as_fast_and_cheap():
    """The faster an agent failed, the better it looked.

    Performance is latency plus cost, and a task that errored returns instantly
    and costs nothing -- so a run whose every task failed reported performance
    1.0 and pulled the global average up with it. Speed cannot be measured on an
    execution that never happened.
    """
    from app.evaluation.metrics.performance import PerformanceMetric
    from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput

    task = EvaluationTaskInput(task_id="x", benchmark="b", category="c", difficulty="easy",
                               domain="travel", prompt="p", expected_answer="a")

    def run(response, errors):
        execution = EvaluationExecutionInput(
            task_id="x", category="c", response=response, latency_seconds=0.001,
            cost_usd=0.0, tool_calls=[], tokens={}, memory_state=[],
            retrieval_documents=[], reasoning_nodes=[], errors=errors)
        return PerformanceMetric().evaluate(task, execution, {})

    crashed = run("", ["API key required for Gemini Developer API"])
    assert crashed.measured is False

    # A genuinely fast successful run is still scored, and scored well.
    succeeded = run("the answer", [])
    assert succeeded.measured is True
    assert succeeded.score > 0.9
