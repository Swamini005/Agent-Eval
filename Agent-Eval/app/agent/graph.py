from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from app.agent.state import AgentState
from app.agent.nodes import (
    intent_detection_node,
    planner_node,
    tool_selection_node,
    execute_tool_node,
    reasoning_node,
    response_generation_node
)

# Define routing logic
def route_after_tool_selection(state: AgentState) -> str:
    """Determine whether to execute tools or skip to reasoning/response."""
    tool_calls = state.get("tool_calls", [])
    if tool_calls:
        return "execute_tool"
    return "reasoning"

def route_after_reasoning(state: AgentState) -> str:
    """Determine whether to run next step in the plan or generate final response."""
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    
    # If there are steps remaining in the plan, loop back to tool selection
    if current_step < len(plan):
        return "tool_selection"
    return "response_generation"

# Build the LangGraph State Machine
builder = StateGraph(AgentState)

# Add all modular nodes
builder.add_node("intent_detection", intent_detection_node)
builder.add_node("planner", planner_node)
builder.add_node("tool_selection", tool_selection_node)
builder.add_node("execute_tool", execute_tool_node)
builder.add_node("reasoning_node", reasoning_node)
builder.add_node("response_generation", response_generation_node)

# Set up edges
builder.add_edge(START, "intent_detection")
builder.add_edge("intent_detection", "planner")
builder.add_edge("planner", "tool_selection")

# Conditional edge from tool selection
builder.add_conditional_edges(
    "tool_selection",
    route_after_tool_selection,
    {
        "execute_tool": "execute_tool",
        "reasoning": "reasoning_node"
    }
)

# Edge from execute tool goes to reasoning
builder.add_edge("execute_tool", "reasoning_node")

# Conditional edge from reasoning (loop or end plan)
builder.add_conditional_edges(
    "reasoning_node",
    route_after_reasoning,
    {
        "tool_selection": "tool_selection",
        "response_generation": "response_generation"
    }
)

builder.add_edge("response_generation", END)

# Compile with in-memory checkpointer
memory = MemorySaver()
travel_agent_graph = builder.compile(checkpointer=memory)
