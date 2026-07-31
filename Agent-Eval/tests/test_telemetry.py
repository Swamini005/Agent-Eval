import pytest
from app.telemetry import telemetry_tracker, LangfuseTracker
from app.benchmarks.runner import BenchmarkRunner
from app.benchmarks.models import UnifiedBenchmarkTask
from app.adapters.factory import AgentFactory

def test_telemetry_tracker_mock_behavior():
    tracker = LangfuseTracker()
    assert tracker.enabled is False
    assert tracker.client is None
    
    # Trace Creation mock check
    trace = tracker.create_trace(
        task_id="t-1",
        benchmark_name="harbor",
        category="flights",
        difficulty="easy",
        prompt="Search flight JFK to LHR"
    )
    
    assert "trace_id" in trace
    assert "deep_link" in trace
    assert "traces/trace_" in trace["deep_link"]
    
    # Callback check under mock mode
    handler = tracker.get_callback_handler(trace["trace_id"])
    assert handler is None

def test_runner_integration_telemetry(tmp_path):
    runner = BenchmarkRunner(
        lambda task_id: AgentFactory.create_agent("langgraph"),
        concurrency=1,
        max_retries=1
    )
    
    task = UnifiedBenchmarkTask(
        id="tel_t1",
        benchmark="contextbench",
        category="weather",
        domain="travel",
        difficulty="medium",
        prompt="Check weather in Paris",
        expected_answer="Cloudy",
        expected_tools=["get_weather"],
        ground_truth={},
        metadata={}
    )
    
    report = runner.run_benchmark([task], output_dir=str(tmp_path))
    assert report["summary"]["successful_runs"] == 1
    
    task_res = report["tasks"][0]
    assert "langfuse_trace_id" in task_res
    assert "langfuse_deep_link" in task_res
    assert "traces/trace_" in task_res["langfuse_deep_link"]
