from typing import Dict, Type, List
from app.evaluation.metrics.base import BaseMetricPlugin

class MetricRegistry:
    """
    Registry for evaluation metric plugins.
    """
    _registry: Dict[str, Type[BaseMetricPlugin]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a custom evaluation metric plugin class."""
        def decorator(subclass: Type[BaseMetricPlugin]):
            subclass.name = name.lower()
            cls._registry[name.lower()] = subclass
            return subclass
        return decorator

    @classmethod
    def get_all_metrics(cls) -> List[BaseMetricPlugin]:
        """Instantiate and return all registered metric plugin objects."""
        return [subclass() for subclass in cls._registry.values()]
