from typing import Dict, Any
from app.evaluation.metrics.base import BaseMetricPlugin
from app.evaluation.metrics.registry import MetricRegistry
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput, MetricResult

@MetricRegistry.register("fault_metrics")
class FaultMetrics(BaseMetricPlugin):
    """
    Computes Fault Detection Rate and Regression Severity.
    """
    
    def evaluate(
        self,
        task: EvaluationTaskInput,
        execution: EvaluationExecutionInput,
        fault_report: Dict[str, Any]
    ) -> MetricResult:
        injected = fault_report.get("injections", [])
        
        # 1. Fault Detection Rate
        # Did the execution report errors or failures when faults were injected?
        triggered_count = len(injected)
        detected_count = 0
        
        for inj in injected:
            # If a critical or error fault was injected and we caught errors or exceptions
            if inj.get("severity") in ["critical", "error", "warning"] and execution.errors:
                detected_count += 1
            # If prompt corruption or bypass was injected and tool calls changed
            elif inj.get("component") == "reasoning" and not execution.tool_calls:
                detected_count += 1
                
        detection_rate = (detected_count / triggered_count) if triggered_count else 1.0
        
        # 2. Regression Severity
        # Severity of performance drop based on injected counts and error instances
        regression_severity = 0.0
        if triggered_count:
            # Scale severity: critical faults count for more
            for inj in injected:
                sev = inj.get("severity", "warning").lower()
                if sev == "critical":
                    regression_severity += 0.4
                elif sev == "error":
                    regression_severity += 0.2
                else:
                    regression_severity += 0.1
            regression_severity = min(1.0, regression_severity)
            
        return MetricResult(
            metric_name="fault_metrics",
            score=round(detection_rate, 3),
            details={
                "fault_detection_rate": round(detection_rate, 3),
                "regression_severity": round(regression_severity, 3),
                "injected_faults_count": triggered_count,
                "detected_faults_count": detected_count
            }
        )
