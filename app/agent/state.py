from typing import TypedDict, List, Dict, Any, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """LangGraph State representation for the Travel Assistant."""
    # List of messages in the conversation (keeps track of chat history)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # Detected intents from user query
    intent: Dict[str, Any]
    
    # List of tasks/plan steps to fulfill the user request
    plan: List[str]
    
    # Track the current plan step index
    current_step: int
    
    # Pending tool calls to execute
    tool_calls: List[Dict[str, Any]]
    
    # Results of executed tools
    tool_results: List[Dict[str, Any]]
    
    # Step-by-step reasoning or agent observations
    reasoning: str
    
    # The generated assistant response to be displayed
    response: str

    # Set by policy_gate_node when the request tries to override pricing, waive a
    # fee or claim authority the sender cannot prove. Non-empty means the agent
    # must refuse: the graph routes straight to the response, and
    # execute_tool_node refuses to run destructive tools regardless of routing.
    policy_refusal: str
