import pytest
from app.adapters.factory import AgentFactory
from app.adapters.base import BaseAgentAdapter
from app.adapters.langgraph import LangGraphTravelAgentAdapter

def test_adapter_factory_and_registration():
    # Verify LangGraph adapter class is registered
    adapter = AgentFactory.create_agent("langgraph")
    assert isinstance(adapter, BaseAgentAdapter)
    assert isinstance(adapter, LangGraphTravelAgentAdapter)

def test_adapter_workflow():
    # Resolve the adapter
    adapter = AgentFactory.create_agent("langgraph")
    
    # Initialize
    adapter.initialize({"session_id": "test_session_123"})
    
    # Verify execution graph structure (nodes, edges)
    exec_graph = adapter.get_execution_graph()
    assert "nodes" in exec_graph
    assert "edges" in exec_graph
    
    node_names = [n["name"] for n in exec_graph["nodes"]]
    assert "intent_detection" in node_names
    assert "planner" in node_names
    assert "tool_selection" in node_names
    assert "execute_tool" in node_names
    assert "reasoning_node" in node_names
    assert "response_generation" in node_names
    
    # Execute a run (using fallback LLM / mock mode)
    task = "Find a flight from JFK to LHR on 2026-08-01 and check weather"
    result = adapter.run(task)
    
    # Check results
    assert "response" in result
    assert "plan" in result
    assert "intent" in result
    
    # Trace
    trace = adapter.get_trace()
    assert len(trace) > 0
    # The last message is usually the final AIMessage response
    assert trace[-1]["content"] == result["response"]
    
    # Tool calls
    tool_calls = adapter.get_tool_calls()
    assert isinstance(tool_calls, list)
    
    # Metrics
    metrics = adapter.get_metrics()
    assert metrics["execution_time_seconds"] > 0
    assert metrics["tool_calls_count"] == len(tool_calls)
    assert metrics["total_messages"] == len(trace)
    
    # Stream test
    adapter.initialize({"session_id": "test_session_stream"})
    steps = list(adapter.stream("Check weather in Tokyo on 2026-09-01"))
    assert len(steps) > 0
    assert "messages" in steps[0]
    
    # Cleanup
    adapter.cleanup()
    assert len(adapter.get_trace()) == 0
    assert len(adapter.get_metrics()) == 0
