import time
import uuid
from typing import Dict, Any, List, Generator, Optional

from app.adapters.base import BaseAgentAdapter
from app.adapters.registry import AdapterRegistry
from app.packs import PackRegistry


@AdapterRegistry.register("react")
class ReActAgentAdapter(BaseAgentAdapter):
    """
    A second, independent agent target built on the plain ReAct loop rather than
    a LangGraph state machine.

    Its purpose is evidential, not functional. A suite written by the same person
    who wrote the agent under test proves nothing on its own -- the checks could
    simply have been shaped around one implementation. Running the identical task
    set against a structurally different agent, and getting a different score,
    is what demonstrates the suite measures the agent rather than describing it.

    It deliberately lacks the LangGraph agent's planner and confirmation
    handling, so it should score worse on plan-dependent and safety-gate tasks.
    That gap is the signal.
    """

    DOMAIN = "travel"

    # Deliberately simple: a single keyword pass over the prompt, no planner and
    # no confirmation stage. This is a weaker agent by construction.
    TOOL_TRIGGERS = {
        "search_flights": ("flight", "fly", "airfare"),
        "search_hotels": ("hotel", "stay", "accommodation"),
        "get_weather": ("weather", "forecast", "temperature"),
        "search_restaurants": ("restaurant", "food", "dine", "eat"),
        "get_route_and_distance": ("route", "drive", "driving", "distance"),
        "convert_currency": ("currency", "exchange", "convert"),
        "get_visa_information": ("visa", "passport"),
        "get_policy_document": ("policy", "refund", "cancel", "change rule", "fee"),
    }

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.session_id: Optional[str] = None
        self.trace: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self.response = ""

    def initialize(self, config: Dict[str, Any]) -> None:
        self.config = dict(config)
        self.session_id = config.get("session_id", f"react_{uuid.uuid4().hex[:8]}")
        self.trace = []
        self.tool_calls = []
        self.response = ""
        self.metrics = {"execution_time_seconds": 0.0, "tool_calls_count": 0, "total_messages": 0}

    def run(self, task: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from app.agent.tools import tools_map

        start = time.time()
        self.trace.append({"type": "HumanMessage", "content": task})

        observations = []
        for tool_name in self._select_tools(task):
            tool = tools_map.get(tool_name)
            if tool is None:
                continue
            args = self._arguments_for(tool_name, task)
            try:
                result = tool.invoke(args)
            except Exception as e:
                # Recorded rather than raised: a tool failure is an observation
                # the suite should score, not a crash of the harness.
                result = f"Tool error: {e}"

            self.tool_calls.append({"tool_name": tool_name, "args": args, "result": result})
            observations.append(f"{tool_name}: {result}")
            self.trace.append({"type": "ToolMessage", "content": str(result)})

        self.response = (
            " ".join(observations) if observations
            else "I could not determine a suitable action for this request."
        )
        self.trace.append({"type": "AIMessage", "content": self.response})

        self.metrics = {
            "execution_time_seconds": round(time.time() - start, 3),
            "tool_calls_count": len(self.tool_calls),
            "total_messages": len(self.trace),
        }

        # No planner: the plan is always empty, which is exactly the capability
        # gap this target exists to expose.
        return {"response": self.response, "plan": [], "intent": {}}

    def stream(self, task: str) -> Generator[Dict[str, Any], None, None]:
        yield self.run(task)

    def get_trace(self) -> List[Dict[str, Any]]:
        return self.trace

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        return self.tool_calls

    def get_retrieval_documents(self) -> List[Dict[str, Any]]:
        pack = PackRegistry.get(self.DOMAIN)
        if pack is None:
            return []
        return [
            {"source": c["tool_name"], "content": str(c["result"]), "args": c["args"]}
            for c in self.tool_calls if c["tool_name"] in pack.retrieval_tools
        ]

    def get_execution_graph(self) -> Dict[str, Any]:
        return {
            "nodes": [{"id": "react_loop", "name": "ReAct Loop"}],
            "edges": []
        }

    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics

    def cleanup(self) -> None:
        self.trace = []
        self.tool_calls = []
        self.metrics = {}
        self.response = ""

    def _select_tools(self, prompt: str) -> List[str]:
        lowered = prompt.lower()
        return [
            tool for tool, triggers in self.TOOL_TRIGGERS.items()
            if any(trigger in lowered for trigger in triggers)
        ]

    def _arguments_for(self, tool_name: str, prompt: str) -> Dict[str, Any]:
        """
        Minimal argument filling.

        No structured extraction, so arguments are frequently wrong. That is a
        genuine weakness of this agent and the suite should penalise it.
        """
        defaults: Dict[str, Dict[str, Any]] = {
            "search_flights": {"origin": "JFK", "destination": "LHR", "date": "2026-09-15"},
            "search_hotels": {"city": "Paris", "checkin_date": "2026-09-15", "checkout_date": "2026-09-18"},
            "get_weather": {"city": "Paris", "date": "2026-09-15"},
            "search_restaurants": {"city": "Paris"},
            "get_route_and_distance": {"origin": "Rome", "destination": "Milan"},
            "convert_currency": {"from_currency": "USD", "to_currency": "EUR"},
            "get_visa_information": {"nationality": "Indian", "destination_country": "France"},
            "get_policy_document": {"policy_type": "refund"},
        }
        return defaults.get(tool_name, {})
