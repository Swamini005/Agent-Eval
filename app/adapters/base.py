from abc import ABC, abstractmethod
from typing import Dict, Any, List, Generator, Optional

class BaseAgentAdapter(ABC):
    """
    Abstract Base Class that defines the common interface for all AI Agent Adapters.
    Allows the evaluation pipeline to evaluate any framework (LangGraph, LangChain, CrewAI, etc.)
    polymorphically.
    """

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize the agent with framework-specific configurations, tools, LLMs, and memory.
        """
        pass

    @abstractmethod
    def run(self, task: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the agent to completion for a given user task.
        Returns the final agent response.
        """
        pass

    @abstractmethod
    def stream(self, task: str) -> Generator[Dict[str, Any], None, None]:
        """
        Stream intermediate updates/steps of the agent execution.
        """
        pass

    @abstractmethod
    def get_trace(self) -> List[Dict[str, Any]]:
        """
        Get the list of raw steps, messages, or state transitions executed during run.
        """
        pass

    @abstractmethod
    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Get the list of tool executions during the run (name, arguments, result).
        """
        pass

    @abstractmethod
    def get_execution_graph(self) -> Dict[str, Any]:
        """
        Get a structural representation of the agent's execution flow (nodes and edges).
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance and usage metrics (execution time, tokens, tools used, etc.).
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Perform teardown actions (e.g. closing connections, cleaning memory).
        """
        pass

    def get_injected_faults(self) -> List[Dict[str, Any]]:
        """
        Get the faults injected during this run, newest last.

        A plain agent injects nothing; decorators such as FaultInjectionMiddleware
        override this. Declaring it here keeps the runner from having to know
        which concrete decorators are in the chain.
        """
        return []

    # Overridden by adapters that know their own tool surface.
    KNOWN_TOOLS = frozenset()

    def capabilities(self):
        """
        What this agent can do, used to skip tasks it cannot attempt.

        The base implementation claims a planner and retrieval, so an adapter
        that has not declared its capabilities runs the whole suite. Skipping
        tasks by accident would quietly shrink the evidence base, which is far
        worse than running more than necessary.
        """
        from app.benchmarks.selection import AgentCapabilities

        return AgentCapabilities(tools=self.KNOWN_TOOLS, plans=True, retrieves=True)

    def get_retrieval_documents(self) -> List[Dict[str, Any]]:
        """
        Get the documents this run actually retrieved, as {source, content}.

        Returns empty for agents that perform no retrieval. Retrieval-quality
        metrics must be able to distinguish "retrieved nothing" from "retrieved
        something", so this reports observed execution only and never synthesises
        placeholder documents.
        """
        return []
