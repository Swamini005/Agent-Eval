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
