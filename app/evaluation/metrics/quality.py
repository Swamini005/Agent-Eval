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
        # 1. Groundedness
        # Calculate matching text characters or overlaps with retrieved documents
        retrieval_text = " ".join(
            doc.get("content", "") for doc in execution.retrieval_documents
        ).lower()
        response_text = execution.response.lower()
        
        words = response_text.split()
        matched_words = [w for w in words if w in retrieval_text] if retrieval_text else words
        groundedness = len(matched_words) / len(words) if words else 1.0
        
        # 2. Hallucination Score (inversely related to groundedness)
        hallucination_score = 1.0 - groundedness
        
        # 3. Reasoning Quality
        # Proportion of completed reasoning nodes/plan items vs planned items
        total_steps = len(execution.reasoning_nodes)
        completed_steps = sum(
            1 for n in execution.reasoning_nodes if n.get("status") == "completed"
        )
        reasoning_quality = completed_steps / total_steps if total_steps else 1.0
        
        # Composite score
        score = (groundedness * 0.4) + ((1.0 - hallucination_score) * 0.3) + (reasoning_quality * 0.3)
        
        return MetricResult(
            metric_name="quality",
            score=round(score, 3),
            details={
                "groundedness": round(groundedness, 3),
                "hallucination_score": round(hallucination_score, 3),
                "reasoning_quality": round(reasoning_quality, 3)
            }
        )
