from typing import Dict, Any
from app.evaluation.metrics.base import BaseMetricPlugin
from app.evaluation.metrics.registry import MetricRegistry
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput, MetricResult

@MetricRegistry.register("memory_and_retrieval")
class MemoryAndRetrievalMetric(BaseMetricPlugin):
    """
    Computes Memory Usage and Retrieval Quality.
    """
    
    def evaluate(
        self,
        task: EvaluationTaskInput,
        execution: EvaluationExecutionInput,
        fault_report: Dict[str, Any]
    ) -> MetricResult:
        # 1. Memory Usage
        # Degradation score if count of message history exceeds a threshold (e.g. 10 messages)
        max_messages = 10
        msg_count = len(execution.memory_state)
        memory_usage_score = max(0.0, 1.0 - (msg_count / max_messages))
        
        # 2. Retrieval Quality
        # Proportion of non-empty documents or scores
        doc_count = len(execution.retrieval_documents)
        retrieval_quality = 1.0 if doc_count > 0 else 0.5
        
        score = (memory_usage_score * 0.5) + (retrieval_quality * 0.5)
        
        return MetricResult(
            metric_name="memory_and_retrieval",
            score=round(score, 3),
            details={
                "memory_usage_score": round(memory_usage_score, 3),
                "message_count": msg_count,
                "retrieval_quality_score": round(retrieval_quality, 3),
                "document_count": doc_count
            }
        )
