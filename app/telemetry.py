import os
import uuid
from typing import Dict, Any, Optional, List
from app.config import settings

try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    CallbackHandler = Any

class LangfuseTracker:
    """
    Unified telemetry manager to trace entire execution pipeline in Langfuse.
    Supports real cloud metrics reporting and offline mock log fallback modes.
    """
    
    def __init__(self):
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY") or settings.metadata.get("LANGFUSE_PUBLIC_KEY") if hasattr(settings, "metadata") else None
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self.project_id = os.getenv("LANGFUSE_PROJECT_ID", "default_proj")
        
        self.client = None
        self.enabled = False
        
        if LANGFUSE_AVAILABLE and self.public_key and self.secret_key:
            try:
                self.client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host
                )
                self.enabled = True
                print("Langfuse telemetry tracing enabled.")
            except Exception as e:
                print(f"Failed to initialize Langfuse client: {e}. Falling back to mock logger.")
        else:
            print("Langfuse credentials missing. Telemetry running in mock mode.")

    def create_trace(
        self,
        task_id: str,
        benchmark_name: str,
        category: str,
        difficulty: str,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Creates a root trace representing a single task evaluation.
        """
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"
        meta_payload = metadata or {}
        meta_payload.update({
            "task_id": task_id,
            "benchmark": benchmark_name,
            "category": category,
            "difficulty": difficulty
        })
        
        if self.enabled and self.client:
            self.client.trace(
                id=trace_id,
                name=f"benchmark_task_{benchmark_name}",
                input=prompt,
                metadata=meta_payload
            )
            
        deep_link = f"{self.host.rstrip('/')}/project/{self.project_id}/traces/{trace_id}"
        return {
            "trace_id": trace_id,
            "deep_link": deep_link
        }

    def get_callback_handler(self, trace_id: str) -> Optional[CallbackHandler]:
        """
        Returns a LangChain-compatible callback handler linked to the trace.
        """
        if self.enabled and LANGFUSE_AVAILABLE:
            return CallbackHandler(
                public_key=self.public_key,
                secret_key=self.secret_key,
                host=self.host,
                trace_id=trace_id
            )
        return None

    def log_evaluation(
        self,
        trace_id: str,
        score: float,
        tool_coverage: float,
        errors: List[str],
        injected_faults: List[Dict[str, Any]]
    ) -> None:
        """
        Logs post-run evaluations, scores, and injected fault metadata to Langfuse.
        """
        if self.enabled and self.client:
            try:
                # Log evaluation scores
                self.client.score(
                    trace_id=trace_id,
                    name="evaluation_score",
                    value=score,
                    comment=f"Tool coverage: {tool_coverage}"
                )
                
                # Update trace with faults and error lists
                self.client.trace(
                    id=trace_id,
                    output={
                        "errors": errors,
                        "injected_faults": injected_faults
                    }
                )
            except Exception as e:
                print(f"Error logging Langfuse score: {e}")
        else:
            print(f"[Mock Log] Trace {trace_id} evaluated. Score: {score}, Tool Coverage: {tool_coverage}, Faults Injected: {len(injected_faults)}")

    def log_span(
        self,
        trace_id: str,
        span_name: str,
        span_input: Any,
        span_output: Any,
        latency: float
    ) -> None:
        """
        Records a span representing an execution block or state machine node.
        """
        if self.enabled and self.client:
            try:
                span = self.client.span(
                    trace_id=trace_id,
                    name=span_name,
                    input=span_input,
                    output=span_output
                )
                # Automatically closes
            except Exception as e:
                print(f"Error logging Langfuse span: {e}")
        else:
            print(f"[Mock Log] Trace {trace_id} span '{span_name}' executed in {latency:.3f}s")
            
# Global telemetry tracker instance
telemetry_tracker = LangfuseTracker()
