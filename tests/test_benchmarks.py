import pytest
from app.benchmarks.registry import BenchmarkRegistry
from app.benchmarks.filters import BenchmarkFilters
from app.benchmarks.dispatcher import BenchmarkDispatcher
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

def test_benchmark_filtering():
    harbor_provider = BenchmarkRegistry.get_provider("harbor")
    cb_provider = BenchmarkRegistry.get_provider("contextbench")
    
    tasks = harbor_provider.load_tasks(HARBOR_DATA) + cb_provider.load_tasks(CONTEXT_DATA)
    assert len(tasks) == 2
    
    # Filter single harbor
    harbor_only = BenchmarkFilters.filter_by_benchmarks(tasks, "harbor")
    assert len(harbor_only) == 1
    assert harbor_only[0].benchmark == "harbor"
    
    # Filter multiple benchmarks
    both = BenchmarkFilters.filter_by_benchmarks(tasks, ["harbor", "contextbench"])
    assert len(both) == 2
    
    # Shuffle and Sample
    shuffled = BenchmarkFilters.shuffle(tasks, seed=42)
    assert len(shuffled) == 2
    
    sampled = BenchmarkFilters.sample(tasks, 1)
    assert len(sampled) == 1

def test_benchmark_dispatcher():
    # Resolve agent adapter
    adapter = AgentFactory.create_agent("langgraph")
    dispatcher = BenchmarkDispatcher(adapter)
    
    harbor_provider = BenchmarkRegistry.get_provider("harbor")
    custom_provider = BenchmarkRegistry.get_provider("custom_json")
    
    tasks = harbor_provider.load_tasks(HARBOR_DATA) + custom_provider.load_tasks(CUSTOM_DATA)
    
    # Dispatch and run evaluations
    report = dispatcher.dispatch(
        tasks=tasks,
        filter_benchmarks=["harbor"],
        shuffle_tasks=False,
        sample_n=1
    )
    
    assert "summary" in report
    assert report["summary"]["total_tasks_attempted"] == 1
    assert report["summary"]["successful_executions"] == 1
    assert len(report["results"]) == 1
    assert report["results"][0]["benchmark"] == "harbor"
    assert "tool_coverage" in report["results"][0]
