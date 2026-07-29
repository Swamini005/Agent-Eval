from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class FaultConfig(BaseModel):
    """
    Configuration format for describing a specific fault injection instruction.
    """
    id: str = Field(..., description="Unique code for this fault rule")
    type: str = Field(..., description="The fault behavior type (e.g. planner_bypass, tool_latency)")
    component: str = Field(..., description="Target component (e.g. reasoning, tool, memory, prompt)")
    severity: str = Field("warning", description="Severity level: info, warning, error, critical")
    probability: float = Field(1.0, description="Activation probability (0.0 to 1.0)")
    scheduling: Dict[str, Any] = Field(default_factory=dict, description="Schedule conditions, e.g., {'after_steps': 1}")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Custom properties like latencies or text overrides")

class FaultLogEntry(BaseModel):
    """
    Represents an occurrence of a triggered fault.
    """
    fault_id: str
    severity: str
    component: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    expected_impact: str
    actual_impact: str
    status: str = "triggered"
    task_id: Optional[str] = None

