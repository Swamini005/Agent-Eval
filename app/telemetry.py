import logging
import os
import uuid
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

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
        # Precedence matters here: the original expression bound as
        # `(getenv(...) or ...) if hasattr(...) else None`, and Settings has no
        # `metadata` attribute, so public_key was always None and telemetry never
        # enabled regardless of the environment.
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
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
                logger.info("Langfuse tracing enabled.")
            except Exception as e:
                logger.warning("Langfuse client init failed (%s); tracing disabled.", e)
        else:
            logger.debug("No Langfuse credentials; tracing disabled.")

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
                logger.warning("Langfuse score not recorded: %s", e)
        else:
            logger.debug("Trace %s scored %.3f (tool coverage %.3f, %d faults)",
                         trace_id, score, tool_coverage, len(injected_faults))

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
                self.client.span(
                    trace_id=trace_id,
                    name=span_name,
                    input=span_input,
                    output=span_output
                )
                # Automatically closes
            except Exception as e:
                logger.warning("Langfuse span not recorded: %s", e)
        else:
            logger.debug("Trace %s span %s took %.3fs", trace_id, span_name, latency)
            
# Global telemetry tracker instance
telemetry_tracker = LangfuseTracker()
