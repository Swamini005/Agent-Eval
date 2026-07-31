import os
import json
import pytest
from app.evaluation.comparator import BaselineComparator

# Mock Summary Payloads
MOCK_SUMMARY_BASE = {
    "summary_metrics": {
        "accuracy": 0.85,
        "tool_accuracy": 0.90,
        "fault_metrics": 0.80
    }
}

MOCK_SUMMARY_CANDIDATE_GOOD = {
    "summary_metrics": {
        "accuracy": 0.84, # minor drop < 0.05
        "tool_accuracy": 0.90,
        "fault_metrics": 0.80
    }
}

MOCK_SUMMARY_CANDIDATE_BAD = {
    "summary_metrics": {
        "accuracy": 0.70, # drop > 0.05
        "tool_accuracy": 0.90,
        "fault_metrics": 0.80
    }
}

MOCK_AGENT_BASE = {
    "agent_performance": {
        "average_latency_seconds": 2.0,
        "average_cost_usd": 0.001
    }
}

MOCK_AGENT_CANDIDATE_GOOD = {
    "agent_performance": {
        "average_latency_seconds": 2.1, # increase < 20%
        "average_cost_usd": 0.001
    }
}

MOCK_AGENT_CANDIDATE_BAD = {
    "agent_performance": {
        "average_latency_seconds": 3.5, # increase > 20%
        "average_cost_usd": 0.001
    }
}

def test_comparator_success(tmp_path):
    comparator = BaselineComparator()
    
    passed = comparator.compare(
        baseline_summary=MOCK_SUMMARY_BASE,
        candidate_summary=MOCK_SUMMARY_CANDIDATE_GOOD,
        baseline_agent_report=MOCK_AGENT_BASE,
        candidate_agent_report=MOCK_AGENT_CANDIDATE_GOOD,
        output_dir=str(tmp_path)
    )
    
    assert passed is True
    
    # Check file exists
    json_file = os.path.join(tmp_path, "comparison.json")
    md_file = os.path.join(tmp_path, "comparison.md")
    assert os.path.exists(json_file)
    assert os.path.exists(md_file)
    
    with open(json_file, "r") as f:
        data = json.load(f)
        assert data["passed"] is True
        assert data["metrics"]["Accuracy"]["status"] == "PASS"

def test_comparator_regression_fails(tmp_path):
    comparator = BaselineComparator()
    
    passed = comparator.compare(
        baseline_summary=MOCK_SUMMARY_BASE,
        candidate_summary=MOCK_SUMMARY_CANDIDATE_BAD,
        baseline_agent_report=MOCK_AGENT_BASE,
        candidate_agent_report=MOCK_AGENT_CANDIDATE_BAD,
        output_dir=str(tmp_path)
    )
    
    assert passed is False
    
    json_file = os.path.join(tmp_path, "comparison.json")
    with open(json_file, "r") as f:
        data = json.load(f)
        assert data["passed"] is False
        assert data["metrics"]["Accuracy"]["status"] == "FAIL"
        assert data["metrics"]["Latency"]["status"] == "FAIL"
