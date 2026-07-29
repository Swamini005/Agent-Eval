from typing import Dict, Any
from app.evaluation.metrics.base import BaseMetricPlugin
from app.evaluation.metrics.registry import MetricRegistry
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput, MetricResult

@MetricRegistry.register("performance")
class PerformanceMetric(BaseMetricPlugin):
    """
    Evaluates execution speed and token cost parameters.
    """
    
    def evaluate(
        self,
        task: EvaluationTaskInput,
        execution: EvaluationExecutionInput,
        fault_report: Dict[str, Any]
    ) -> MetricResult:
        # Normalize latency: score degrades as latency approaches 15.0 seconds
        max_latency = 15.0
        lat_score = max(0.0, 1.0 - (execution.latency_seconds / max_latency))
        
        # Normalize cost: score degrades as cost approaches $0.05
        max_cost = 0.05
        cost_score = max(0.0, 1.0 - (execution.cost_usd / max_cost))
        
        overall_perf = (lat_score * 0.6) + (cost_score * 0.4)
        
        return MetricResult(
            metric_name="performance",
            score=round(overall_perf, 3),
            details={
                "latency_score": round(lat_score, 3),
                "cost_score": round(cost_score, 3),
                "latency_seconds": execution.latency_seconds,
                "cost_usd": execution.cost_usd
            }
        )
