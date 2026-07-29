from typing import Dict, Any
from app.evaluation.metrics.base import BaseMetricPlugin
from app.evaluation.metrics.registry import MetricRegistry
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput, MetricResult

@MetricRegistry.register("tool_accuracy")
class ToolAccuracyMetric(BaseMetricPlugin):
    """
    Computes precision, recall, and overlap ratio of tools executed against expected tool calls.
    """
    
    def evaluate(
        self,
        task: EvaluationTaskInput,
        execution: EvaluationExecutionInput,
        fault_report: Dict[str, Any]
    ) -> MetricResult:
        expected = set(task.expected_tools)
        actual = {call.get("tool_name") for call in execution.tool_calls if call.get("tool_name")}
        
        if not expected and not actual:
            return MetricResult(metric_name="tool_accuracy", score=1.0, details={"precision": 1.0, "recall": 1.0})
            
        if not expected and actual:
            return MetricResult(metric_name="tool_accuracy", score=0.0, details={"precision": 0.0, "recall": 0.0, "unwanted_tools": list(actual)})
            
        # Compute precision and recall
        intersection = expected.intersection(actual)
        precision = len(intersection) / len(actual) if actual else 0.0
        recall = len(intersection) / len(expected) if expected else 0.0
        
        # F1 score as normalized metric value
        score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return MetricResult(
            metric_name="tool_accuracy",
            score=round(score, 3),
            details={
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "expected_tools": list(expected),
                "actual_tools": list(actual)
            }
        )
