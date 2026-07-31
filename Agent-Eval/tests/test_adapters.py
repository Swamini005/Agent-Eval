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


# --- foreign agent shapes --------------------------------------------------
#
# The suite's central claim is that it evaluates any agent from any framework.
# Each shape below was found broken against a live third-party agent: the two
# that returned an unresolved coroutine or generator are the dangerous ones,
# because str() of either is a response-shaped string, so the pipeline scored an
# agent that never executed and reported it as the agent's failure.

import asyncio
from dataclasses import dataclass, field

import pytest

from app.adapters.external import (
    call_agent, extract_response, extract_tool_calls, instantiate, resolve,
)


def test_async_agent_is_awaited_not_returned_unrun():
    class Agent:
        async def run(self, prompt):
            return {"response": f"answered {prompt}"}

    result = call_agent(Agent(), "hi")
    assert extract_response(result) == "answered hi"


def test_async_generator_agent_is_drained():
    class Agent:
        def run(self, prompt):
            async def chunks():
                for word in ("one ", "two ", "three"):
                    yield word
            return chunks()

    assert extract_response(call_agent(Agent(), "hi")) == "one two three"


def test_sync_generator_agent_is_drained():
    class Agent:
        def run(self, prompt):
            yield "a"
            yield "b"

    assert extract_response(call_agent(Agent(), "hi")) == "ab"


def test_agent_requiring_dict_input_is_accommodated():
    """The LangChain AgentExecutor convention: invoke() takes a dict."""
    class Agent:
        def invoke(self, payload):
            if not isinstance(payload, dict):
                raise TypeError("Input must be a dict")
            return {"output": f"answered {payload['input']}"}

    assert extract_response(call_agent(Agent(), "hi")) == "answered hi"


def test_agent_requiring_messages_input_is_accommodated():
    class Agent:
        def invoke(self, state):
            if not isinstance(state, dict) or "messages" not in state:
                raise TypeError("expects messages")
            return {"messages": state["messages"] + [{"role": "assistant", "content": "done"}]}

    assert extract_response(call_agent(Agent(), "hi")) == "done"


def test_response_is_read_from_a_message_list():
    """LangGraph and LCEL return the conversation, not an answer field."""
    result = {"messages": [{"role": "user", "content": "q"},
                           {"role": "assistant", "content": "the answer"}]}
    assert extract_response(result) == "the answer"


def test_response_and_tools_are_read_off_a_dataclass():
    """Falling through to str() yields "AgentResult(final_output=...)", which
    looks like an answer and scores like one."""
    @dataclass
    class AgentResult:
        final_output: str
        tool_calls: list = field(default_factory=list)

    result = AgentResult(final_output="the answer", tool_calls=[{"name": "get_weather"}])

    assert extract_response(result) == "the answer"
    assert [c["tool_name"] for c in extract_tool_calls(result)] == ["get_weather"]


def test_tool_calls_are_read_off_a_message_object():
    from langchain_core.messages import AIMessage

    message = AIMessage(content="done")
    message.tool_calls = [{"name": "search_flights", "args": {"origin": "BOM"}, "id": "1"}]

    assert extract_response(message) == "done"
    assert [c["tool_name"] for c in extract_tool_calls(message)] == ["search_flights"]


def test_gemini_style_block_content_is_flattened():
    """Gemini returns typed blocks; str() of the list is not an answer."""
    assert extract_response({"response": [{"type": "text", "text": "hello"}]}) == "hello"


def test_a_class_path_is_instantiated():
    """Handing over a class is the normal case; unbound, the prompt binds to self."""
    class Agent:
        def run(self, prompt):
            return "ok"

    assert isinstance(instantiate(Agent), Agent)
    assert call_agent(instantiate(Agent), "hi") == "ok"


def test_a_class_needing_constructor_arguments_says_so_clearly():
    class Agent:
        def __init__(self, api_key):
            self.api_key = api_key

    with pytest.raises(ValueError, match="constructor arguments"):
        instantiate(Agent)


def test_an_agent_reporting_no_tools_is_not_recorded_as_having_used_none():
    """None and [] differ: unobservable is not the same as observed-empty."""
    assert extract_tool_calls("a plain string answer") is None
    assert extract_tool_calls({"response": "no tool key at all"}) is None
    assert extract_tool_calls({"response": "x", "tool_calls": []}) == []


def test_an_agent_that_accepts_nothing_names_the_shapes_tried():
    class Agent:
        def run(self, prompt, config):
            return "never reached"

    with pytest.raises(TypeError, match="neither a string"):
        call_agent(Agent(), "hi")
