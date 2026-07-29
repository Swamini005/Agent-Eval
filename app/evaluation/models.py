from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class EvaluationTaskInput(BaseModel):
    task_id: str
    benchmark: str
    category: str = "general"
    prompt: str
    expected_answer: Optional[str] = None
    expected_tools: List[str] = Field(default_factory=list)

class EvaluationExecutionInput(BaseModel):
    task_id: str
    category: str = "general"
    response: str
    latency_seconds: float
    cost_usd: float
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tokens: Dict[str, int] = Field(default_factory=dict)
    memory_state: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_documents: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

class MetricResult(BaseModel):
    metric_name: str
    score: float = Field(..., description="Normalized score between 0.0 and 1.0")
    details: Dict[str, Any] = Field(default_factory=dict)
