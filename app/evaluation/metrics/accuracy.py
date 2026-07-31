from typing import Dict, Any
from app.evaluation.metrics.base import BaseMetricPlugin
from app.evaluation.metrics.registry import MetricRegistry
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput, MetricResult

@MetricRegistry.register("accuracy")
class AccuracyMetric(BaseMetricPlugin):
    """
    Lexical overlap between the response and a reference answer.

    This is a proxy, not a correctness judgement, and it is sensitive to
    phrasing: a right answer worded differently from the reference scores near
    zero. It therefore flatters an agent whose phrasing the reference answers
    were written alongside, and understates any other. Read it next to the
    assertion and tool metrics, which check contracts rather than wording.
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

        # No reference answer means there is nothing to compare against, which is
        # not the same as a wrong answer -- and scoring it anyway inverts the
        # result. `target in actual` is trivially true for an empty target, so a
        # task with no expected answer scored 0.4 for arbitrary output and a full
        # 1.0 when the agent returned nothing at all.
        if not target:
            return MetricResult(
                metric_name="accuracy",
                score=0.0,
                measured=False,
                details={"reason": "task declares no expected_answer"},
            )

        # An agent that produced nothing has not answered. `actual in target` is
        # trivially true for an empty string, so a task that errored scored a
        # full exact match and 0.4 overall -- which is how a run whose every task
        # failed still reported a global average of 0.549.
        if not actual:
            return MetricResult(
                metric_name="accuracy",
                score=0.0,
                details={"exact_match": False, "reason": "agent returned no response"},
            )

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
