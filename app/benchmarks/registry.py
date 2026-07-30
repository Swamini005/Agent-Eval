from typing import Dict, List, Type
from app.benchmarks.base import BaseBenchmarkProvider

class BenchmarkRegistry:
    """
    Registry to catalog concrete Benchmark Providers.
    """
    _registry: Dict[str, Type[BaseBenchmarkProvider]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a concrete benchmark provider by name."""
        def decorator(subclass: Type[BaseBenchmarkProvider]):
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get_provider(cls, name: str) -> BaseBenchmarkProvider:
        """Instantiate and retrieve the provider registered under a name."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            raise KeyError(
                f"Benchmark Provider '{name}' is not registered. Available providers: {list(cls._registry.keys())}"
            )
        return cls._registry[name_lower]()

    @classmethod
    def available(cls) -> List[str]:
        """Names of every registered provider, for error messages and discovery."""
        return sorted(cls._registry)
