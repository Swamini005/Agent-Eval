from typing import Dict, Callable, Any
from app.faults.models import FaultConfig, FaultLogEntry

class FaultRegistry:
    """
    Registry for fault mutation functions.
    Allows mapping string names to logic blocks.
    """
    _registry: Dict[str, Callable[[Any, FaultConfig], Any]] = {}

    @classmethod
    def register(cls, fault_type: str):
        """Decorator to register a custom fault mutation action by type name."""
        def decorator(func: Callable[[Any, FaultConfig], Any]):
            cls._registry[fault_type.lower()] = func
            return func
        return decorator

    @classmethod
    def get_handler(cls, fault_type: str) -> Callable[[Any, FaultConfig], Any]:
        """Gets the handler function for a given fault type."""
        fault_type_lower = fault_type.lower()
        if fault_type_lower not in cls._registry:
            # Fallback identity handler
            return lambda obj, config: obj
        return cls._registry[fault_type_lower]
