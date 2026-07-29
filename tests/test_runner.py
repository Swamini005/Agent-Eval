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
    adapter = AgentFactory.create_agent("langgraph")
    runner = BenchmarkRunner(adapter, concurrency=2, max_retries=1)
    
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
    
    runner = BenchmarkRunner(failed_adapter, concurrency=1, max_retries=2)
    
    # Run task
    result = runner._execute_task_with_retries(TASKS_RAW[0])
    
    # Verify failed run metadata
    assert result["success"] is False
    assert result["attempts"] == 3 # Initial attempt (1) + 2 retries = 3
    assert len(result["errors"]) == 3
    assert "Model rate limit exceeded" in result["errors"][0]
