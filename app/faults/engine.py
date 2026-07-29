import json
import os
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.faults.models import FaultConfig, FaultLogEntry

class FaultInjectionEngine:
    """
    Orchestration engine for checking scheduling, evaluating probabilities,
    and recording triggered faults to JSON logs and reports.
    """
    
    def __init__(self, config_rules: List[FaultConfig] = None):
        self.configs: List[FaultConfig] = config_rules or []
        self.logs: List[FaultLogEntry] = []
        self.step_counter = 0
        self.current_task_id: Optional[str] = None

    def trigger_fault(self, fault_type: str, component: str) -> Optional[FaultConfig]:
        """
        Evaluate if a fault should be injected.
        Checks type, component, probability, and schedules.
        """
        for config in self.configs:
            if config.type.lower() == fault_type.lower() and config.component.lower() == component.lower():
                # Roll probability
                if random.random() > config.probability:
                    continue
                    
                # Evaluate scheduling (e.g. after_steps, step_counter matches)
                after_steps = config.scheduling.get("after_steps", 0)
                if self.step_counter < after_steps:
                    continue
                    
                return config
        return None

    def log_trigger(self, config: FaultConfig, expected_impact: str, actual_impact: str) -> None:
        """Record the triggered fault event."""
        entry = FaultLogEntry(
            fault_id=config.id,
            severity=config.severity,
            component=config.component,
            expected_impact=expected_impact,
            actual_impact=actual_impact,
            task_id=self.current_task_id
        )
        self.logs.append(entry)
        print(f"FAULT INJECTED [{config.severity.upper()}]: {config.type} on {config.component}. Expected: {expected_impact}")

    def increment_steps(self) -> None:
        self.step_counter += 1

    def save_reports(self, workspace_path: str = ".") -> None:
        """
        Generates fault_log.json and fault_report.json inside the workspace.
        """
        log_file = os.path.join(workspace_path, "fault_log.json")
        report_file = os.path.join(workspace_path, "fault_report.json")
        
        serialized_logs = [log.model_dump() for log in self.logs]
        
        # 1. Save fault_log.json
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(serialized_logs, f, indent=2)
            
        # 2. Summarize and save fault_report.json
        severity_counts = {}
        component_counts = {}
        for entry in self.logs:
            severity_counts[entry.severity] = severity_counts.get(entry.severity, 0) + 1
            component_counts[entry.component] = component_counts.get(entry.component, 0) + 1
            
        report = {
            "summary": {
                "total_faults_injected": len(self.logs),
                "timestamp": datetime.utcnow().isoformat(),
                "severity_distribution": severity_counts,
                "component_distribution": component_counts
            },
            "injections": serialized_logs
        }
        
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        print(f"Fault logs saved to {log_file} and {report_file}")
