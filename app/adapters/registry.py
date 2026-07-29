from typing import Dict, Type
from app.adapters.base import BaseAgentAdapter

class AdapterRegistry:
    """
    Registry to dynamically catalog and look up concrete Agent Adapters.
    """
    _registry: Dict[str, Type[BaseAgentAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a concrete adapter class by name."""
        def decorator(subclass: Type[BaseAgentAdapter]):
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get_adapter_class(cls, name: str) -> Type[BaseAgentAdapter]:
        """Retrieve the adapter class registered under a given name."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise KeyError(
                f"Adapter '{name}' is not registered. Available adapters: {list(cls._registry.keys())}"
            )
        return cls._registry[name_lower]
