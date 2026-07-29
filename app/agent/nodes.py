import json
from typing import Dict, Any, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings
from app.agent.state import AgentState
from app.agent.tools import travel_tools, tools_map

# Enable global in-memory caching for LLMs
set_llm_cache(InMemoryCache())


def get_llm() -> BaseChatModel:
    """Helper to initialize the LLM based on configuration."""
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "google":
        model_name = settings.MODEL_NAME or "gemini-1.5-flash"
        # API key is automatically picked up from GEMINI_API_KEY / GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    elif provider == "openai":
        model_name = settings.MODEL_NAME or "gpt-4o-mini"
        return ChatOpenAI(model=model_name, temperature=0, api_key=settings.OPENAI_API_KEY)
    else:
        # Fallback/Mock LLM if none configured or invalid
        raise ValueError(f"Unsupported LLM provider: {provider}")

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
        response = llm.invoke([SystemMessage(content="You are an intent detection module."), HumanMessage(content=prompt)])
        
        # Clean response content and parse JSON
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
            
        intent_data = json.loads(content)
    except Exception as e:
        # Fallback parsing in case of API failure or parser issue
        print(f"Error in intent detection: {e}. Using rule-based fallback.")
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
        response = llm.invoke([SystemMessage(content="You are a travel planner module."), HumanMessage(content=prompt)])
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        elif content.startswith("```"):
            content = content.replace("```", "").strip()
        plan = json.loads(content)
    except Exception as e:
        print(f"Error in planner node: {e}. Generating default plan.")
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
        response = llm_with_tools.invoke([
            SystemMessage(content="You are a tool selector module. Choose the single best tool and arguments for the current step."),
            HumanMessage(content=prompt)
        ])
        
        tool_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                tool_calls.append({
                    "name": tc["name"],
                    "args": tc["args"],
                    "id": tc["id"]
                })
        return {"tool_calls": tool_calls}
    except Exception as e:
        print(f"Error in tool selection: {e}. Choosing no tool/fallback.")
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
        
        print(f"Executing tool: {tool_name} with args {tool_args}")
        
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
    print(reasoning_summary)
    
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
        response = llm.invoke([
            SystemMessage(content="You are a professional travel assistant. Synthesize all details into a structured, highly useful response."),
            HumanMessage(content=prompt)
        ])
        final_text = response.content
    except Exception as e:
        # Fallback text generator
        print(f"Error in response generation: {e}")
        final_text = "Here is the summary of your travel details:\n\n"
        for tr in tool_results:
            final_text += f"### {tr['tool_name'].replace('_', ' ').title()}\n{tr['result']}\n\n"
            
    return {
        "response": final_text,
        "messages": [AIMessage(content=final_text)]
    }
