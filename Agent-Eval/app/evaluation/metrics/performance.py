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
        # A task that crashed returns almost instantly and costs nothing, which
        # scored a perfect 1.0 -- so the faster an agent failed, the better it
        # looked. A run whose every task errored reported performance 1.0 and
        # lifted the global average with it. Speed is not measurable on an
        # execution that never happened, so it is excluded from the aggregate
        # rather than counted as either a pass or a failure.
        if execution.errors and not (execution.response or "").strip():
            return MetricResult(
                metric_name="performance",
                score=0.0,
                measured=False,
                details={"reason": "execution failed; latency reflects the failure, "
                                   "not the agent",
                         "errors": execution.errors[:3]},
            )

        # Normalize latency. A live model is two orders of magnitude slower than
        # the rule-based path, so one fixed budget would score every real run
        # near zero and drag the gate down for a reason that says nothing about
        # the agent. Mirrors the split in gate_thresholds.yaml.
        from app.config import settings

        max_latency = 15.0 if settings.LLM_PROVIDER.lower() == "mock" else 60.0
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
