from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class UnifiedBenchmarkTask(BaseModel):
    """
    Standardized schema for benchmark tasks across all providers.
    Ensures evaluation pipelines receive uniform data structures.
    """
    id: str = Field(..., description="Unique identifier for the task")
    benchmark: str = Field(..., description="Name of the source benchmark (e.g. harbor, contextbench)")
    category: str = Field(..., description="Category of the task (e.g. flight_booking, multi_city)")
    domain: str = Field(..., description="Application domain (e.g. travel, general)")
    difficulty: str = Field(..., description="Task difficulty level (e.g. easy, medium, hard)")
    prompt: str = Field(..., description="The query/instruction provided to the agent")
    expected_answer: Optional[str] = Field(None, description="Correct target text response if available")
    expected_tools: List[str] = Field(default_factory=list, description="List of expected tool names to be executed")
    ground_truth: Optional[Dict[str, Any]] = Field(None, description="Detailed ground truth properties for assertions")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary provider-specific task metadata")
