import time
import uuid
from typing import Dict, Any, List, Generator, Optional
from langchain_core.messages import HumanMessage
from app.adapters.base import BaseAgentAdapter
from app.adapters.registry import AdapterRegistry
from app.agent.graph import travel_agent_graph
from app.agent.nodes import collected_usage, reset_usage
from app.packs import PackRegistry

@AdapterRegistry.register("langgraph")
class LangGraphTravelAgentAdapter(BaseAgentAdapter):
    """
    Adapter implementing the BaseAgentAdapter interface for the LangGraph-based AI Travel Assistant.
    """

    # Domain whose pack declares which of this agent's tools perform retrieval.
    DOMAIN = "travel"

    def __init__(self):
        self.graph = None
        self.config = {}
        self.last_state = {}
        self.trace: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {}
        self.session_id = None

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the LangGraph instance. 
        Accepts settings like custom session_id or uses default compiled graph.
        """
        # Token counts accumulate per run, so the previous task's usage must not
        # be attributed to this one.
        reset_usage()

        self.graph = travel_agent_graph
        self.session_id = config.get("session_id", f"adapter_{uuid.uuid4().hex[:8]}")
        self.config = {"configurable": {"thread_id": self.session_id}}
        self.trace = []
        self.metrics = {
            "execution_time_seconds": 0.0,
            "tool_calls_count": 0,
            "total_messages": 0
        }

    def run(self, task: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Executes the LangGraph assistant synchronously.
        """
        if not self.graph:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")
            
        start_time = time.time()
        
        run_config = dict(self.config)
        if config:
            run_config.update({k: v for k, v in config.items() if k != "configurable"})
            if "configurable" in config:
                run_config["configurable"] = {**self.config.get("configurable", {}), **config["configurable"]}
        
        # Invoke state machine
        state = self.graph.invoke(
            {"messages": [HumanMessage(content=task)]},
            config=run_config
        )
        
        end_time = time.time()
        
        self.last_state = state
        self._process_execution(state, end_time - start_time)
        
        return {
            "response": state.get("response", ""),
            "plan": state.get("plan", []),
            "intent": state.get("intent", {})
        }

    def stream(self, task: str) -> Generator[Dict[str, Any], None, None]:
        """
        Streams updates from the LangGraph assistant execution.
        """
        if not self.graph:
            raise RuntimeError("Adapter not initialized. Call initialize() first.")
            
        start_time = time.time()
        
        stream_generator = self.graph.stream(
            {"messages": [HumanMessage(content=task)]},
            config=self.config,
            stream_mode="values"
        )
        
        state = {}
        for update in stream_generator:
            state = update
            yield {
                "messages": [m.content for m in update.get("messages", [])],
                "response": update.get("response", ""),
                "current_step": update.get("current_step", 0)
            }
            
        end_time = time.time()
        self.last_state = state
        self._process_execution(state, end_time - start_time)

    def get_trace(self) -> List[Dict[str, Any]]:
        """
        Returns full detailed step trace (messages) of the last run.
        """
        return self.trace

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Collects details of all tool executions.
        """
        return self.last_state.get("tool_results", [])

    def capabilities(self):
        """Every registered tool, and a real planner stage."""
        from app.agent.tools import tools_map
        from app.benchmarks.selection import AgentCapabilities

        return AgentCapabilities(tools=set(tools_map), plans=True, retrieves=True)

    def get_retrieval_documents(self) -> List[Dict[str, Any]]:
        """
        Surfaces documents fetched by this agent's knowledge-retrieval tools.

        Which tools count as retrieval comes from the domain pack. Derived from
        observed tool results, so a run that retrieved nothing reports nothing,
        and a run served corrupted context reports the corrupted content it
        actually received.
        """
        pack = PackRegistry.get(self.DOMAIN)
        if pack is None:
            return []

        return [
            {
                "source": call.get("tool_name"),
                "content": str(call.get("result", "")),
                "args": call.get("args", {})
            }
            for call in self.get_tool_calls()
            if call.get("tool_name") in pack.retrieval_tools
        ]

    def get_execution_graph(self) -> Dict[str, Any]:
        """
        Returns the nodes and edges of the compiled LangGraph state machine.
        """
        if not self.graph:
            return {"nodes": [], "edges": []}
            
        graph_data = self.graph.get_graph()
        nodes = [{"id": n.id, "name": n.name} for n in graph_data.nodes.values()]
        edges = [{"source": e.source, "target": e.target} for e in graph_data.edges]
        
        return {
            "nodes": nodes,
            "edges": edges
        }

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return timing and execution counts.
        """
        return self.metrics

    def cleanup(self) -> None:
        """
        Clears stored run state.
        """
        self.last_state = {}
        self.trace = []
        self.metrics = {}

    def _process_execution(self, state: Dict[str, Any], duration: float) -> None:
        """Helper to compute execution metrics and traces from state."""
        messages = state.get("messages", [])
        
        # Populate Trace
        self.trace = []
        for msg in messages:
            self.trace.append({
                "type": type(msg).__name__,
                "content": msg.content,
                "additional_kwargs": getattr(msg, "additional_kwargs", {})
            })
            
        # Compute metrics
        tool_results = state.get("tool_results", [])
        self.metrics = {
            "execution_time_seconds": round(duration, 3),
            "tool_calls_count": len(tool_results),
            "total_messages": len(messages),
            # Present only when the provider reported usage. Absent in
            # deterministic mode, where the runner falls back to an estimate and
            # labels it as such.
            **collected_usage(),
        }
