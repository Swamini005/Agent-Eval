from abc import ABC, abstractmethod
from typing import Dict, Any
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput, MetricResult

class BaseMetricPlugin(ABC):
    """
    Abstract Base Class for Metric Plugins.
    New metrics can be added to the pipeline without modifying evaluation core code.
    """
    
    @abstractmethod
    def evaluate(
        self,
        task: EvaluationTaskInput,
        execution: EvaluationExecutionInput,
        fault_report: Dict[str, Any]
    ) -> MetricResult:
        """
        Calculates a score and returns details for a task execution.
        """
        pass
