from abc import ABC, abstractmethod
from typing import List, Any
from app.benchmarks.models import UnifiedBenchmarkTask

class BaseBenchmarkProvider(ABC):
    """
    Abstract Base Class that defines the normalization interface for benchmark providers.
    Allows dynamic loading and formatting of external benchmark datasets into UnifiedBenchmarkTask.
    """
    
    @abstractmethod
    def load_tasks(self, raw_data: Any) -> List[UnifiedBenchmarkTask]:
        """
        Normalize framework-specific raw task data into unified task models.
        """
        pass
