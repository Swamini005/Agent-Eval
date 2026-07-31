from typing import Dict, Any
from app.evaluation.metrics.base import BaseMetricPlugin
from app.evaluation.metrics.registry import MetricRegistry
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput, MetricResult

@MetricRegistry.register("quality")
class QualityMetric(BaseMetricPlugin):
    """
    Computes Groundedness, Hallucination Score, and Reasoning Quality.
    """
    
    def evaluate(
        self,
        task: EvaluationTaskInput,
        execution: EvaluationExecutionInput,
        fault_report: Dict[str, Any]
    ) -> MetricResult:
        # 1. Groundedness -- share of response tokens supported by retrieved text.
        #
        # Only meaningful when the agent actually retrieved something. A run with
        # no retrieval is not "perfectly grounded"; it is unmeasurable, and
        # scoring it 1.0 would silently reward agents that skip retrieval
        # entirely. Such runs are reported as unmeasured and excluded from the
        # composite rather than assigned a flattering default.
        retrieval_text = " ".join(
            doc.get("content", "") for doc in execution.retrieval_documents
        ).lower()
        words = execution.response.lower().split()

        grounded = bool(retrieval_text) and bool(words)
        if grounded:
            matched_words = [w for w in words if w in retrieval_text]
            groundedness = len(matched_words) / len(words)
            hallucination_score = 1.0 - groundedness
        else:
            groundedness = None
            hallucination_score = None

        # 2. Reasoning Quality -- share of planned steps that were carried out.
        total_steps = len(execution.reasoning_nodes)
        completed_steps = sum(
            1 for n in execution.reasoning_nodes if n.get("status") == "completed"
        )
        reasoning_quality = completed_steps / total_steps if total_steps else 0.0

        # Composite. Groundedness and hallucination are the same signal --
        # hallucination is defined as its complement -- so weighting both double
        # counts one measurement. Groundedness carries it once.
        if grounded:
            score = (groundedness * 0.6) + (reasoning_quality * 0.4)
        else:
            score = reasoning_quality

        return MetricResult(
            metric_name="quality",
            score=round(score, 3),
            details={
                "groundedness": round(groundedness, 3) if grounded else None,
                "hallucination_score": round(hallucination_score, 3) if grounded else None,
                "reasoning_quality": round(reasoning_quality, 3),
                "retrieval_available": grounded
            }
        )
