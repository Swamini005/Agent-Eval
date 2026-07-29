from typing import Dict, Any, Optional
from app.adapters.registry import AdapterRegistry
from app.adapters.base import BaseAgentAdapter

class AgentFactory:
    """
    Factory Pattern with Dependency Injection to resolve and initialize agent adapters.
    """
    @staticmethod
    def create_agent(
        framework_name: str,
        config: Optional[Dict[str, Any]] = None,
        llm: Optional[Any] = None
    ) -> BaseAgentAdapter:
        """
        Creates, configures, and initializes an agent adapter.
        Injects dependencies such as custom configurations and LLMs.
        """
        config = config or {}
        
        # Inject LLM if provided
        if llm:
            config["llm"] = llm
            
        adapter_cls = AdapterRegistry.get_adapter_class(framework_name)
        adapter_instance = adapter_cls()
        adapter_instance.initialize(config)
        return adapter_instance
