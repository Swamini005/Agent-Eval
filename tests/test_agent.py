import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.services.mocks import (
    mock_flight_search,
    mock_hotel_search,
    mock_weather_forecast,
    mock_maps_route,
    mock_currency_conversion,
    mock_restaurant_search
)
from app.agent.state import AgentState
from app.agent.llm import DeterministicMode
from app.agent.nodes import (
    intent_detection_node,
    planner_node,
    tool_selection_node,
    execute_tool_node,
    reasoning_node,
    response_generation_node
)
from app.agent.graph import route_after_tool_selection, route_after_reasoning

# Test Mock Services
def test_mock_services():
    # Test Flight
    flights = mock_flight_search("JFK", "LAX", "2026-08-01")
    assert len(flights) > 0
    assert flights[0]["origin"] == "JFK"
    
    # Test Hotel
    hotels = mock_hotel_search("Paris", "2026-08-01", "2026-08-05", "luxury")
    assert len(hotels) > 0
    assert hotels[0]["category"] == "luxury"
    
    # Test Weather
    weather = mock_weather_forecast("Tokyo", "2026-08-01")
    assert weather["city"] == "Tokyo"
    assert "temperature_celsius" in weather
    
    # Test Maps
    route = mock_maps_route("New York", "Boston", "driving")
    assert route["origin"] == "New York"
    assert route["distance_km"] > 0
    
    # Test Currency
    conv = mock_currency_conversion("USD", "EUR", 100.0)
    assert conv["from"] == "USD"
    assert conv["converted_amount"] > 0
    
    # Test Restaurants
    restaurants = mock_restaurant_search("Rome", "Italian")
    assert len(restaurants) > 0
    assert restaurants[0]["cuisine"] == "Italian"

# Test Graph Nodes using Mocked LLM or Fallbacks
@patch("app.agent.nodes.get_llm")
def test_intent_detection_fallback(mock_get_llm):
    # Deterministic mode selects the rule-based path
    mock_get_llm.side_effect = DeterministicMode("LLM_PROVIDER=mock")
    
    state: AgentState = {
        "messages": [HumanMessage(content="Search a flight to Paris and find a nice hotel")],
        "intent": {},
        "plan": [],
        "current_step": 0,
        "tool_calls": [],
        "tool_results": [],
        "reasoning": "",
        "response": ""
    }
    
    res = intent_detection_node(state)
    assert res["intent"]["flights"] is True
    assert res["intent"]["hotels"] is True
    assert res["intent"]["weather"] is False

@patch("app.agent.nodes.get_llm")
def test_planner_node_fallback(mock_get_llm):
    mock_get_llm.side_effect = DeterministicMode("LLM_PROVIDER=mock")
    
    state: AgentState = {
        "messages": [HumanMessage(content="Check weather in London")],
        "intent": {"weather": True},
        "plan": [],
        "current_step": 0,
        "tool_calls": [],
        "tool_results": [],
        "reasoning": "",
        "response": ""
    }
    
    res = planner_node(state)
    assert len(res["plan"]) > 0
    assert "Check weather forecast" in res["plan"]

def test_execute_tool_node():
    state: AgentState = {
        "messages": [],
        "intent": {},
        "plan": [],
        "current_step": 0,
        "tool_calls": [
            {
                "name": "get_weather",
                "args": {"city": "Tokyo", "date": "2026-08-01"},
                "id": "call_123"
            }
        ],
        "tool_results": [],
        "reasoning": "",
        "response": ""
    }
    
    res = execute_tool_node(state)
    assert len(res["tool_results"]) == 1
    assert res["tool_results"][0]["tool_name"] == "get_weather"
    assert "Tokyo" in res["tool_results"][0]["result"]
    assert len(res["messages"]) == 1
    assert isinstance(res["messages"][0], ToolMessage)
    # Ensure pending tool calls are cleared
    assert len(res["tool_calls"]) == 0

def test_reasoning_node():
    state: AgentState = {
        "messages": [],
        "intent": {},
        "plan": ["Step 1", "Step 2"],
        "current_step": 0,
        "tool_calls": [],
        "tool_results": [],
        "reasoning": "",
        "response": ""
    }
    
    res = reasoning_node(state)
    assert res["current_step"] == 1
    assert "Step 1" in res["reasoning"]

def test_routing_functions():
    # route_after_tool_selection
    state_with_calls = {"tool_calls": [{"name": "get_weather", "args": {}, "id": "1"}]}
    state_without_calls = {"tool_calls": []}
    
    assert route_after_tool_selection(state_with_calls) == "execute_tool"
    assert route_after_tool_selection(state_without_calls) == "reasoning"
    
    # route_after_reasoning
    state_not_done = {"plan": ["step1", "step2"], "current_step": 1}
    state_done = {"plan": ["step1", "step2"], "current_step": 2}
    
    assert route_after_reasoning(state_not_done) == "tool_selection"
    assert route_after_reasoning(state_done) == "response_generation"
