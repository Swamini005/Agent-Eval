import os
import json
import pytest
from unittest.mock import MagicMock
from app.benchmarks.runner import BenchmarkRunner
from app.benchmarks.models import UnifiedBenchmarkTask
from app.adapters.factory import AgentFactory

# Helper raw task dataset
TASKS_RAW = [
    UnifiedBenchmarkTask(
        id="runner_t1",
        benchmark="harbor",
        category="flight",
        domain="travel",
        difficulty="easy",
        prompt="Search flight JFK to LHR on 2026-09-01",
        expected_answer="SkyFlow TX-101",
        expected_tools=["search_flights"],
        ground_truth={},
        metadata={}
    ),
    UnifiedBenchmarkTask(
        id="runner_t2",
        benchmark="harbor",
        category="weather",
        domain="travel",
        difficulty="medium",
        prompt="Check weather forecast in Paris",
        expected_answer="Partly Cloudy",
        expected_tools=["get_weather"],
        ground_truth={},
        metadata={}
    )
]

def test_benchmark_runner_success(tmp_path):
    runner = BenchmarkRunner(
        lambda task_id: AgentFactory.create_agent("langgraph"),
        concurrency=2,
        max_retries=1
    )

    # Run parallel execution
    report = runner.run_benchmark(TASKS_RAW, output_dir=str(tmp_path))
    
    # Check return report structure
    assert "summary" in report
    assert report["summary"]["total_tasks"] == 2
    assert report["summary"]["successful_runs"] == 2
    
    # Verify execution.json file output
    execution_file = os.path.join(tmp_path, "execution.json")
    assert os.path.exists(execution_file)
    
    with open(execution_file, "r") as f:
        data = json.load(f)
        assert data["summary"]["total_tasks"] == 2
        assert len(data["tasks"]) == 2
        
        task_1 = data["tasks"][0]
        assert "prompt" in task_1
        assert "response" in task_1
        assert "tool_calls" in task_1
        assert "execution_graph" in task_1
        assert "latency_seconds" in task_1
        assert "cost_usd" in task_1
        assert "tokens" in task_1
        assert "errors" in task_1
        assert "memory_state" in task_1
        assert "retrieval_documents" in task_1
        assert "reasoning_nodes" in task_1

def test_runner_retry_logic():
    # Setup mock adapter that always throws exceptions
    failed_adapter = MagicMock()
    failed_adapter.run.side_effect = Exception("Model rate limit exceeded")

    runner = BenchmarkRunner(lambda task_id: failed_adapter, concurrency=1, max_retries=2)

    # Run task
    result = runner._execute_task_with_retries(TASKS_RAW[0])

    # Verify failed run metadata
    assert result["success"] is False
    assert result["attempts"] == 3 # Initial attempt (1) + 2 retries = 3
    assert len(result["errors"]) == 3
    assert "Model rate limit exceeded" in result["errors"][0]


def test_runner_isolates_adapter_per_task():
    """Each task must receive its own adapter instance, never a shared one."""
    handed_out = []

    def factory(task_id):
        adapter = MagicMock()
        adapter.run.return_value = {"response": "ok", "plan": [], "intent": {}}
        adapter.get_trace.return_value = []
        adapter.get_tool_calls.return_value = []
        adapter.get_execution_graph.return_value = {"nodes": [], "edges": []}
        adapter.get_metrics.return_value = {}
        adapter.get_injected_faults.return_value = []
        adapter.get_retrieval_documents.return_value = []
        handed_out.append((task_id, adapter))
        return adapter

    runner = BenchmarkRunner(factory, concurrency=2, max_retries=0)
    for task in TASKS_RAW:
        runner._execute_task_with_retries(task)

    assert [task_id for task_id, _ in handed_out] == ["runner_t1", "runner_t2"]
    assert handed_out[0][1] is not handed_out[1][1]


def test_runner_reports_only_documents_the_agent_retrieved():
    """Retrieval documents come from the adapter, never synthesised by the runner."""
    adapter = MagicMock()
    adapter.run.return_value = {"response": "ok", "plan": [], "intent": {}}
    adapter.get_trace.return_value = []
    adapter.get_tool_calls.return_value = []
    adapter.get_execution_graph.return_value = {"nodes": [], "edges": []}
    adapter.get_metrics.return_value = {}
    adapter.get_injected_faults.return_value = []
    adapter.get_retrieval_documents.return_value = []

    result = BenchmarkRunner(lambda task_id: adapter, max_retries=0)._execute_task_with_retries(TASKS_RAW[0])

    assert result["retrieval_documents"] == []
