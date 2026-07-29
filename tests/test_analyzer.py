import os
import json
import pytest
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
