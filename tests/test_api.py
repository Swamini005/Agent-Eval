import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_api_chat():
    payload = {
        "message": "Find flights from JFK to LAX on 2026-09-01",
        "session_id": "test_chat_session"
    }
    res = client.post("/api/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "response" in data
    assert "plan" in data

def test_api_benchmark_run():
    payload = {
        "tasks": [
            {
                "id": "api_t1",
                "benchmark": "harbor",
                "category": "flight",
                "domain": "travel",
                "difficulty": "easy",
                "prompt": "Find a flight from LHR to JFK on 2026-08-01",
                "expected_answer": "SkyFlow TX-101",
                "expected_tools": ["search_flights"],
                "metadata": {}
            }
        ],
        "faults": [
            {
                "id": "F-API-01",
                "type": "tool_latency",
                "component": "tool",
                "severity": "info",
                "probability": 1.0,
                "scheduling": {},
                "parameters": {"delay_seconds": 0.05}
            }
        ],
        "concurrency": 2,
        "max_retries": 1
    }
    
    res = client.post("/api/benchmark/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    
    # Assert output formatting
    assert "summary" in data
    assert "results" in data
    
    summary = data["summary"]
    assert summary["total_tasks_evaluated"] == 1
    assert "global_average_score" in summary
    assert "summary_metrics" in summary
    assert "execution_summary" in summary
    assert "fault_summary" in summary
    
    # Detailed result
    task_res = data["results"][0]
    assert task_res["task_id"] == "api_t1"
    assert "overall_score" in task_res
    assert "metrics" in task_res
    assert "details" in task_res
