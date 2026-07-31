import os
import json
import pytest
import tempfile
from app.evaluation.analyzer import FailureAnalyzer

def test_failure_analyzer_wrong_tool(tmp_path):
    history_file = os.path.join(tmp_path, "history.json")
    analyzer = FailureAnalyzer(history_file=history_file)
    
    results = [
        {
            "task_id": "t1",
            "overall_score": 0.50,
            "metrics": {
                "tool_accuracy": 0.20 # low tool accuracy
            }
        }
    ]
    
    executions = [
        {
            "task_id": "t1",
            "errors": [],
            "latency_seconds": 1.5,
            "tool_calls": []
        }
    ]
    
    fault_report = {"injections": []}
    
    failures = analyzer.analyze(results, executions, fault_report, output_dir=str(tmp_path))
    assert len(failures) == 1
    assert failures[0]["category"] == "Wrong Tool"
    assert failures[0]["historical_frequency"] == 1
    assert "suggested_fix" in failures[0]
    
    report_file = os.path.join(tmp_path, "failure_report.json")
    assert os.path.exists(report_file)
    with open(report_file, "r") as f:
        data = json.load(f)
        assert data["total_failures_detected"] == 1

def test_failure_analyzer_hallucination(tmp_path):
    history_file = os.path.join(tmp_path, "history.json")
    analyzer = FailureAnalyzer(history_file=history_file)
    
    results = [
        {
            "task_id": "t2",
            "overall_score": 0.40,
            "metrics": {},
            "details": {
                "quality": {
                    "hallucination_score": 0.85 # high hallucination
                }
            }
        }
    ]
    
    executions = [
        {
            "task_id": "t2",
            "errors": [],
            "latency_seconds": 1.0,
            "tool_calls": []
        }
    ]
    
    failures = analyzer.analyze(results, executions, {}, output_dir=str(tmp_path))
    assert len(failures) == 1
    assert failures[0]["category"] == "Hallucination"


def test_hallucination_branch_survives_unmeasurable_groundedness():
    """quality.hallucination_score is None when the run retrieved nothing.

    dict.get(key, default) returns None for a present-but-None key, so comparing
    it against a float raises TypeError. The analyzer must fall through instead.
    """
    analyzer = FailureAnalyzer(history_file=os.devnull)
    result = {
        "task_id": "t1",
        "overall_score": 0.5,
        "metrics": {"tool_accuracy": 1.0, "memory_and_retrieval": 1.0, "quality": 1.0},
        "details": {"quality": {"hallucination_score": None, "groundedness": None}},
    }
    execution = {"task_id": "t1", "category": "general", "errors": [], "latency_seconds": 0.1,
                 "retrieval_documents": [], "memory_state": [], "reasoning_nodes": []}

    diagnosis = analyzer._diagnose_failure(result, execution, injected_faults=[])

    assert diagnosis["category"] != "Hallucination"


def test_diagnosis_considers_every_fault_injected_into_a_task():
    """A task can receive several faults; the analyzer must see all of them.

    Indexing injections into a dict keyed on task_id silently keeps only the
    last one, hiding the safety-critical fault behind whichever was logged last.
    """
    analyzer = FailureAnalyzer(history_file=os.devnull)
    results = [{"task_id": "t1", "overall_score": 0.2, "metrics": {}, "details": {}}]
    executions = [{"task_id": "t1", "category": "general", "errors": [],
                   "latency_seconds": 0.1, "retrieval_documents": [],
                   "memory_state": [], "reasoning_nodes": []}]
    fault_report = {"injections": [
        {"task_id": "t1", "type": "planner_bypass_confirmation", "component": "reasoning"},
        {"task_id": "t1", "type": "tool_latency", "component": "tool"},
    ]}

    failures = analyzer.analyze(results, executions, fault_report, output_dir=str(tempfile.mkdtemp()))

    assert failures[0]["fault_type"] == "planner_bypass_confirmation"
