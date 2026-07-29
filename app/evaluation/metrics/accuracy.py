from typing import Dict, Any
from app.evaluation.metrics.base import BaseMetricPlugin
from app.evaluation.metrics.registry import MetricRegistry
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput, MetricResult

@MetricRegistry.register("accuracy")
class AccuracyMetric(BaseMetricPlugin):
    """
    Computes exact match accuracy and semantic similarity overlap.
    """
    
    def evaluate(
        self,
        task: EvaluationTaskInput,
        execution: EvaluationExecutionInput,
        fault_report: Dict[str, Any]
    ) -> MetricResult:
        # 1. Exact Match Check
        target = (task.expected_answer or "").strip().lower()
        actual = (execution.response or "").strip().lower()
        
        exact_match = 1.0 if target in actual or actual in target else 0.0
        
        # 2. Semantic Similarity Overlap
        # Let's compute Jaccard similarity of words
        target_words = set(target.split())
        actual_words = set(actual.split())
        
        jaccard = 1.0
        if target_words or actual_words:
            intersection = target_words.intersection(actual_words)
            union = target_words.union(actual_words)
            jaccard = len(intersection) / len(union) if union else 1.0
            
        score = (exact_match * 0.4) + (jaccard * 0.6)
        
        return MetricResult(
            metric_name="accuracy",
            score=round(score, 3),
            details={
                "exact_match": exact_match == 1.0,
                "jaccard_similarity": round(jaccard, 3)
            }
        )
