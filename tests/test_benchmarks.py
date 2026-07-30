import pytest
from app.benchmarks.registry import BenchmarkRegistry
from app.benchmarks.models import UnifiedBenchmarkTask
from app.adapters.factory import AgentFactory

# Mock dataset payloads
HARBOR_DATA = [
    {
        "task_id": "harbor_t1",
        "category": "flight_search",
        "domain": "travel",
        "difficulty_level": "easy",
        "task_prompt": "Find a flight from JFK to LHR on 2026-08-01",
        "target_response": "SkyFlow TX-101 departs JFK at 08:00.",
        "required_tools": ["search_flights"],
        "validation_rules": {"exact_match": True}
    }
]

CONTEXT_DATA = [
    {
        "id": "cb_t1",
        "context_type": "weather_lookup",
        "task_domain": "weather",
        "complexity": "hard",
        "query": "Check weather forecast in Rome",
        "expected_output": "Weather is Partly Cloudy, 25C.",
        "tools_to_call": ["get_weather"],
        "ground_truth_data": {"threshold": 2.0}
    }
]

T3_DATA = [
    {
        "t3_id": "t3_t1",
        "scenario": "multicity_route",
        "application": "maps",
        "level": "medium",
        "instruction": "Find driving route from Paris to Berlin",
        "answer_keys": "Distance: 1050 km",
        "tool_sequence": ["get_route_and_distance"]
    }
]

CUSTOM_DATA = [
    {
        "id": "custom_t1",
        "benchmark": "test_suite",
        "category": "restaurant_dine",
        "domain": "dining",
        "difficulty": "easy",
        "query": "Find Italian restaurants in Tokyo",
        "expected_answer": "The Golden Spoon",
        "expected_tools": ["search_restaurants"]
    }
]

def test_normalization_and_registry():
    # Retrieve and run Harbor normalization
    harbor_provider = BenchmarkRegistry.get_provider("harbor")
    normalized_harbor = harbor_provider.load_tasks(HARBOR_DATA)
    assert len(normalized_harbor) == 1
    assert isinstance(normalized_harbor[0], UnifiedBenchmarkTask)
    assert normalized_harbor[0].benchmark == "harbor"
    assert normalized_harbor[0].prompt == HARBOR_DATA[0]["task_prompt"]
    assert normalized_harbor[0].expected_tools == ["search_flights"]

    # ContextBench
    cb_provider = BenchmarkRegistry.get_provider("contextbench")
    normalized_cb = cb_provider.load_tasks(CONTEXT_DATA)
    assert normalized_cb[0].benchmark == "contextbench"
    assert normalized_cb[0].difficulty == "hard"
    assert normalized_cb[0].expected_tools == ["get_weather"]

    # T3Bench
    t3_provider = BenchmarkRegistry.get_provider("t3bench")
    normalized_t3 = t3_provider.load_tasks(T3_DATA)
    assert normalized_t3[0].benchmark == "t3bench"
    assert normalized_t3[0].domain == "maps"
    assert normalized_t3[0].expected_tools == ["get_route_and_distance"]

    # Custom JSON
    custom_provider = BenchmarkRegistry.get_provider("custom_json")
    normalized_custom = custom_provider.load_tasks(CUSTOM_DATA)
    assert normalized_custom[0].benchmark == "test_suite"
    assert normalized_custom[0].expected_answer == "The Golden Spoon"
    assert normalized_custom[0].expected_tools == ["search_restaurants"]


def test_foreign_benchmark_exports_load_through_a_provider(tmp_path, monkeypatch):
    """A raw Harbor-format export can be dropped into tasks/ and evaluated.

    This is what the provider registry is for: task files are normally already
    in unified shape, but an external benchmark dump should not have to be
    rewritten by hand before it can be run.
    """
    import json
    from app.benchmarks import suites

    monkeypatch.setattr(suites, "TASKS_DIR", str(tmp_path))
    (tmp_path / "imported.json").write_text(json.dumps([
        {
            "task_id": "h-1",
            "benchmark": "harbor",
            "category": "flight_planning",
            "domain": "travel",
            "difficulty_level": "easy",
            "task_prompt": "Find a flight from JFK to LHR",
            "target_response": "TX-101",
            "required_tools": ["search_flights"],
        }
    ]), encoding="utf-8")

    suite = suites.load_suite("imported")

    assert len(suite) == 1
    assert suite.tasks[0].id == "h-1"
    assert suite.tasks[0].prompt == "Find a flight from JFK to LHR"
    assert suite.tasks[0].expected_tools == ["search_flights"]
    assert suite.sha


def test_registry_lists_its_providers():
    assert set(BenchmarkRegistry.available()) >= {"harbor", "contextbench", "t3bench", "custom_json"}
