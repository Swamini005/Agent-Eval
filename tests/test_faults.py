import os
import pytest
from app.adapters.factory import AgentFactory
from app.faults.models import FaultConfig
from app.faults.engine import FaultInjectionEngine
from app.faults.middleware import FaultInjectionMiddleware
from app.faults.loader import FaultConfigLoader

# Mock faults configuration mapping
FAULTS_RAW = {
    "faults": [
        {
            "id": "F-01",
            "type": "prompt_corruption",
            "component": "prompt",
            "severity": "warning",
            "probability": 1.0,
            "scheduling": {},
            "parameters": {}
        },
        {
            "id": "F-02",
            "type": "planner_bypass",
            "component": "reasoning",
            "severity": "critical",
            "probability": 1.0,
            "scheduling": {"after_steps": 0},
            "parameters": {}
        },
        {
            "id": "F-03",
            "type": "tool_latency",
            "component": "tool",
            "severity": "info",
            "probability": 1.0,
            "scheduling": {},
            "parameters": {"delay_seconds": 0.1}
        }
    ]
}

def test_fault_config_loader():
    configs = FaultConfigLoader.load_from_dict(FAULTS_RAW["faults"])
    assert len(configs) == 3
    assert configs[0].id == "F-01"
    assert configs[0].type == "prompt_corruption"
    assert configs[1].severity == "critical"

def test_fault_injection_workflow(tmp_path):
    # Resolve the base agent adapter
    base_adapter = AgentFactory.create_agent("langgraph")
    
    # Load configs and initialize Engine + Middleware
    configs = FaultConfigLoader.load_from_dict(FAULTS_RAW["faults"])
    engine = FaultInjectionEngine(configs)
    
    middleware = FaultInjectionMiddleware(base_adapter, engine)
    
    # 1. Initialize middleware (checks model config/API faults)
    middleware.initialize({"session_id": "test_faults_session"})
    
    # 2. Run middleware (trigger prompt corruption, tool latency, planner bypass)
    task = "Find a flight from London to Tokyo on 2026-09-01"
    res = middleware.run(task)
    
    # 3. Assertions
    # Planner bypass should wipe out the plan list
    assert len(res.get("plan", [])) == 0
    
    # Trace/memory check
    trace = middleware.get_trace()
    assert len(trace) > 0
    
    # 4. Save and verify JSON reports
    middleware.engine.save_reports(workspace_path=str(tmp_path))
    
    log_file = os.path.join(tmp_path, "fault_log.json")
    report_file = os.path.join(tmp_path, "fault_report.json")
    
    assert os.path.exists(log_file)
    assert os.path.exists(report_file)
    
    # Check log contents
    import json
    with open(log_file, "r") as f:
        logs = json.load(f)
        assert len(logs) > 0
        triggered_ids = [l["fault_id"] for l in logs]
        assert "F-01" in triggered_ids # prompt corruption
        assert "F-02" in triggered_ids # planner bypass


def test_fork_isolates_state_and_is_deterministic():
    """Forked engines share rules but never share counters, logs, or RNG state."""
    configs = FaultConfigLoader.load_from_dict(FAULTS_RAW["faults"])
    root = FaultInjectionEngine(configs, seed=42)

    a, b = root.fork("task-a"), root.fork("task-b")
    assert a is not b
    assert a.current_task_id == "task-a" and b.current_task_id == "task-b"

    a.increment_steps()
    assert b.step_counter == 0, "step counters must not be shared between tasks"

    # Same root seed + same task id must reproduce the same random stream.
    again = FaultInjectionEngine(configs, seed=42).fork("task-a")
    assert [a._random.random() for _ in range(5)] == [again._random.random() for _ in range(5)]

    # The root aggregates every child's logs for reporting.
    fault = a.trigger_fault("prompt_corruption", "prompt")
    a.log_trigger(fault, "expected", "actual")
    assert [entry.task_id for entry in root.collect_logs()] == ["task-a"]


def test_retry_does_not_duplicate_fault_log_entries():
    """A retried task reports the faults of the attempt that was kept, not every attempt.

    Duplicate entries inflate injections_by_type, which is the denominator of the
    regression catch rate.
    """
    configs = FaultConfigLoader.load_from_dict(FAULTS_RAW["faults"])
    root = FaultInjectionEngine(configs, seed=0)
    adapter = FaultInjectionMiddleware(AgentFactory.create_agent("langgraph"), root.fork("t1"))

    for _ in range(3):
        adapter.initialize({"session_id": "s", "task_id": "t1"})
        adapter.engine.log_trigger(configs[0], "expected", "actual")

    assert len(root.collect_logs()) == 1
