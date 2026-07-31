from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class EvaluationTaskInput(BaseModel):
    task_id: str
    benchmark: str
    category: str = "general"
    difficulty: str = Field(
        "unknown",
        description="Task difficulty. Carried through so reports can break scores "
                    "down by it; it exists on the benchmark task and was being "
                    "dropped at this boundary."
    )
    domain: str = Field(
        "",
        description="Selects the domain pack supplying safety vocabulary for this "
                    "task. Tasks from different domains may share a single run."
    )
    prompt: str
    expected_answer: Optional[str] = None
    expected_tools: List[str] = Field(default_factory=list)
    ground_truth: Optional[Dict[str, Any]] = Field(
        None,
        description="Declarative, machine-checkable assertions for this task. "
                    "See AssertionMetric for the supported clauses."
    )

class EvaluationExecutionInput(BaseModel):
    task_id: str
    category: str = "general"
    response: str
    latency_seconds: float
    cost_usd: float
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tokens: Dict[str, int] = Field(default_factory=dict)
    token_source: str = Field(
        "estimated",
        description="'provider' when the model reported real usage, 'estimated' "
                    "when tokens were inferred from character counts. Cost derived "
                    "from an estimate must not be presented as a measurement."
    )
    memory_state: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_documents: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning_nodes: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

class MetricResult(BaseModel):
    metric_name: str
    score: float = Field(..., description="Normalized score between 0.0 and 1.0")
    measured: bool = Field(
        True,
        description="False when this metric had nothing to evaluate -- no domain "
                    "pack, no assertions declared, no retrieval performed. "
                    "Unmeasured results are excluded from aggregate scores rather "
                    "than counted as either a pass or a failure, because both "
                    "would misreport the agent."
    )
    details: Dict[str, Any] = Field(default_factory=dict)
