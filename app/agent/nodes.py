import logging
import json
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from app.agent.llm import DeterministicMode, get_llm, text_of, token_usage
from app.agent.state import AgentState
from app.agent.tools import travel_tools, tools_map

logger = logging.getLogger(__name__)

# Deliberately NO global LLM cache here.
#
# `set_llm_cache(InMemoryCache())` used to run at import. It is process-global, so
# importing this module silently changed behaviour for every chat model in the
# process -- including third-party agents under test and the triage advisor.
#
# In an evaluation harness that destroys the measurement. Seeds exist to sample a
# stochastic agent; for the clean arm the prompts are identical across seeds, so
# every seed after the first returned the first one's cached response. Five seeds
# became one sample and four copies, stdev was 0 by construction, and the Wilson
# intervals and Fisher tests were computed over pseudo-replicates.
#
# Caching belongs at the run level, where RegressionExperiment does it: keyed on
# seed, arm, provider and suite, so repeats are skipped without collapsing variance.

# Real token counts as reported by the provider, accumulated across the LLM calls
# a single task makes. The runner previously estimated tokens as len(text)//4 and
# priced that estimate as if it were measured; with a real model the provider
# reports exact figures, so the cost it derives is the real cost.
_USAGE: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}


def reset_usage() -> None:
    """Clear the accumulator. Called by the adapter before each run."""
    _USAGE.update({"prompt_tokens": 0, "completion_tokens": 0})


def collected_usage() -> Dict[str, int]:
    """Usage reported so far, or {} when the provider reported none."""
    if not any(_USAGE.values()):
        return {}
    return dict(_USAGE)


def _track(response):
    """Accumulate provider-reported usage; returns the response unchanged."""
    for key, value in token_usage(response).items():
        _USAGE[key] = _USAGE.get(key, 0) + value
    return response


def intent_detection_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to detect user intent (e.g. flights, hotels, weather, currency, budget, multi-city).
    Runs structured prediction to populate intent state.
    """
    last_message = state["messages"][-1].content
    
    # Prompt to extract user intent
    prompt = (
        "Identify the travel assistant capabilities requested in the user query. "
        "Return a JSON object with the following boolean fields indicating what the user is asking for:\n"
        "- flights: true/false\n"
        "- hotels: true/false\n"
        "- attractions: true/false\n"
        "- restaurants: true/false\n"
        "- weather: true/false\n"
        "- currency: true/false\n"
        "- visa: true/false\n"
        "- budget: true/false\n"
        "- multi_city: true/false\n"
        "- general_travel_query: true/false\n\n"
        f"User Query: \"{last_message}\"\n\n"
        "Return ONLY the JSON block. Do not wrap in markdown unless standard json formats."
    )
    
    try:
        llm = get_llm()
        response = _track(llm.invoke([SystemMessage(content="You are an intent detection module."), HumanMessage(content=prompt)]))
        
        # Clean response content and parse JSON
        content = text_of(response).strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        intent_data = json.loads(content)
    except DeterministicMode:
        # Deliberate: no model configured, so use the rule-based path. A real
        # model error is not caught here and fails the task instead.
        query_lower = last_message.lower()
        intent_data = {
            "flights": "flight" in query_lower or "fly" in query_lower or "ticket" in query_lower,
            "hotels": "hotel" in query_lower or "stay" in query_lower or "accommodation" in query_lower,
            "attractions": "attraction" in query_lower or "visit" in query_lower or "see" in query_lower or "places to" in query_lower,
            "restaurants": "restaurant" in query_lower or "food" in query_lower or "eat" in query_lower or "dine" in query_lower,
            "weather": "weather" in query_lower or "forecast" in query_lower or "rain" in query_lower or "temp" in query_lower,
            "currency": "currency" in query_lower or "exchange" in query_lower or "convert" in query_lower or "rate" in query_lower,
            "visa": "visa" in query_lower or "passport" in query_lower or "entry requirement" in query_lower,
            "budget": "budget" in query_lower or "cost" in query_lower or "expensive" in query_lower or "price" in query_lower or "estimate" in query_lower,
            "multi_city": "multi" in query_lower or "itinerary" in query_lower or "trip plan" in query_lower,
            "general_travel_query": True
        }
        
    return {
        "intent": intent_data
    }

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to formulate a high-level step-by-step plan based on user queries and detected intent.
    """
    last_message = state["messages"][-1].content
    intent = state.get("intent", {})
    
    prompt = (
        f"Create a step-by-step sequential plan to answer the user query: \"{last_message}\"\n"
        f"Detected Intents: {json.dumps(intent)}\n\n"
        "Return a JSON list of strings, where each string represents a specific step (e.g., ['Check flight options from X to Y', 'Get weather forecast for Y']).\n"
        "Keep the plan concise and structured. Return ONLY the JSON list of strings."
    )
    
    try:
        llm = get_llm()
        response = _track(llm.invoke([SystemMessage(content="You are a travel planner module."), HumanMessage(content=prompt)]))
        content = text_of(response).strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        plan = json.loads(content)
    except DeterministicMode:
        # Deliberate rule-based path; real model errors propagate.
        plan = []
        if intent.get("flights"):
            plan.append("Search flights matching the query")
        if intent.get("hotels"):
            plan.append("Search hotels matching the query")
        if intent.get("weather"):
            plan.append("Check weather forecast")
        if intent.get("restaurants"):
            plan.append("Search local restaurants")
        if intent.get("visa"):
            plan.append("Retrieve visa entry requirements")
        if intent.get("currency"):
            plan.append("Convert currencies if needed")
        if not plan:
            plan.append("Provide a general helpful travel response")
            
    return {
        "plan": plan,
        "current_step": 0
    }

def tool_selection_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to select the next tool to run based on the current step in the plan and conversation context.
    Uses LLM tool binding.
    """
    plan = state.get("plan", [])
    current_step_idx = state.get("current_step", 0)
    messages = state["messages"]
    
    if current_step_idx >= len(plan):
        # Plan is complete, no tool needed
        return {"tool_calls": []}
        
    current_step_desc = plan[current_step_idx]
    
    prompt = (
        f"Conversation History: {[m.content for m in messages[-3:]]}\n"
        f"We are at step {current_step_idx + 1} of the plan: \"{current_step_desc}\"\n"
        "Select the appropriate tool and arguments to execute this step."
    )
    
    try:
        llm = get_llm()
        llm_with_tools = llm.bind_tools(travel_tools)
        response = _track(llm_with_tools.invoke([
            SystemMessage(content="You are a tool selector module. Choose the single best tool and arguments for the current step."),
            HumanMessage(content=prompt)
        ]))
        
        tool_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append({
                    "name": tc["name"],
                    "args": tc["args"],
                    "id": tc["id"]
                })
        return {"tool_calls": tool_calls}
    except DeterministicMode:
        # Deliberate rule-based path; real model errors propagate.
        return {"tool_calls": []}

def execute_tool_node(state: AgentState) -> Dict[str, Any]:
    """
    Node that runs the selected tool(s) and collects their outputs.
    """
    tool_calls = state.get("tool_calls") or []
    tool_results = list(state.get("tool_results") or [])
    new_messages = []
    
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]
        
        logger.debug("Tool call: %s(%s)", tool_name, tool_args)
        
        if tool_name in tools_map:
            try:
                # Call tool
                tool_output = tools_map[tool_name].invoke(tool_args)
            except Exception as e:
                tool_output = f"Error executing tool {tool_name}: {str(e)}"
        else:
            tool_output = f"Tool {tool_name} not found."
            
        tool_results.append({
            "tool_name": tool_name,
            "args": tool_args,
            "result": tool_output
        })
        
        # Append ToolMessage to keep graph/chat history correct
        new_messages.append(ToolMessage(
            content=str(tool_output),
            name=tool_name,
            tool_call_id=tool_id
        ))
        
    return {
        "tool_results": tool_results,
        "messages": new_messages,
        "tool_calls": [] # Clear pending tool calls
    }

def reasoning_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to review progress, synthesize tool outputs, and decide if another plan step is required.
    """
    plan = state.get("plan") or []
    current_step_idx = state.get("current_step", 0)
    tool_results = state.get("tool_results") or []
    
    # We increment the plan step counter
    next_step_idx = current_step_idx + 1
    
    reasoning_summary = f"Completed step {next_step_idx} of {len(plan)}: {plan[current_step_idx] if current_step_idx < len(plan) else 'N/A'}"
    # One line per plan step per task. It belongs in the trace, which it already
    # is via the return value, not on stdout.
    logger.debug("%s", reasoning_summary)
    
    return {
        "current_step": next_step_idx,
        "reasoning": reasoning_summary
    }

def response_generation_node(state: AgentState) -> Dict[str, Any]:
    """
    Node to formulate the final response to the user.
    """
    messages = state["messages"]
    tool_results = state.get("tool_results") or []
    plan = state.get("plan") or []
    
    # Render final response
    prompt = (
        "Generate a comprehensive, professional, and friendly response answering the user's request. "
        f"Plan executed: {json.dumps(plan)}\n"
        f"Tool execution results: {json.dumps(tool_results)}\n\n"
        "Integrate these details into a clear travel guide or itinerary as requested. "
        "Include budget estimations, flight numbers, hotel ratings, weather conditions, or restaurant specialties if relevant. "
        "Highlight visa requirements clearly if visa info is present."
    )
    
    try:
        llm = get_llm()
        response = _track(llm.invoke([
            SystemMessage(content="You are a professional travel assistant. Synthesize all details into a structured, highly useful response."),
            HumanMessage(content=prompt)
        ]))
        final_text = text_of(response)
    except DeterministicMode:
        # Deliberate rule-based path; real model errors propagate.
        final_text = "Here is the summary of your travel details:\n\n"
        for tr in tool_results:
            final_text += f"### {tr['tool_name'].replace('_', ' ').title()}\n{tr['result']}\n\n"
            
    return {
        "response": final_text,
        "messages": [AIMessage(content=final_text)]
    }
