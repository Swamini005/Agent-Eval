import logging
import json
import os
import random
import threading
import zlib
from typing import List, Dict, Optional
from datetime import datetime, timezone
from app.faults.models import FaultConfig, FaultLogEntry

logger = logging.getLogger(__name__)

class FaultInjectionEngine:
    """
    Orchestration engine for checking scheduling, evaluating probabilities,
    and recording triggered faults to JSON logs and reports.

    An engine owns mutable execution state (step counter, task id, logs), so a
    single instance must never be shared across concurrently executing tasks.
    Callers create one root engine to hold the rule set, then derive a private
    child per task via ``fork``. Children report their logs back to the root,
    which aggregates them for reporting.
    """

    def __init__(
        self,
        config_rules: Optional[List[FaultConfig]] = None,
        seed: Optional[int] = None,
        task_id: Optional[str] = None,
    ):
        self.configs: List[FaultConfig] = config_rules or []
        self.logs: List[FaultLogEntry] = []
        self.step_counter = 0
        self.current_task_id: Optional[str] = task_id
        self.seed = seed
        self._random = random.Random(seed)
        self._children: List["FaultInjectionEngine"] = []
        self._lock = threading.Lock()

    def fork(self, task_id: str) -> "FaultInjectionEngine":
        """
        Derive an isolated engine for a single task execution.

        The child shares the rule set but owns its own step counter, log buffer,
        and random stream. The child's seed is derived deterministically from the
        root seed and the task id, so a given (seed, task_id) pair always
        produces the same injection decisions regardless of scheduling order.
        """
        child_seed = None
        if self.seed is not None:
            child_seed = (self.seed ^ zlib.crc32(task_id.encode("utf-8"))) & 0xFFFFFFFF

        child = FaultInjectionEngine(self.configs, seed=child_seed, task_id=task_id)
        with self._lock:
            self._children.append(child)
        return child

    def collect_logs(self) -> List[FaultLogEntry]:
        """Return this engine's logs plus every forked child's, ordered by task."""
        with self._lock:
            children = list(self._children)
        collected = list(self.logs)
        for child in children:
            collected.extend(child.collect_logs())
        return collected

    def trigger_fault(self, fault_type: str, component: str) -> Optional[FaultConfig]:
        """
        Evaluate if a fault should be injected.
        Checks type, component, probability, and schedules.
        """
        for config in self.configs:
            if config.type.lower() != fault_type.lower():
                continue
            if config.component.lower() != component.lower():
                continue

            # Evaluate scheduling (e.g. after_steps) before spending a random draw,
            # so the random stream stays aligned regardless of when a rule becomes
            # eligible.
            after_steps = config.scheduling.get("after_steps", 0)
            if self.step_counter < after_steps:
                continue

            if self._random.random() > config.probability:
                continue

            return config
        return None

    def log_trigger(self, config: FaultConfig, expected_impact: str, actual_impact: str) -> None:
        """Record the triggered fault event."""
        entry = FaultLogEntry(
            fault_id=config.id,
            type=config.type,
            severity=config.severity,
            component=config.component,
            expected_impact=expected_impact,
            actual_impact=actual_impact,
            task_id=self.current_task_id
        )
        self.logs.append(entry)
        # Per-injection and therefore very high volume: one line per fault per
        # task. Debug so a normal run stays readable and -v still shows it.
        logger.debug("Fault injected [%s]: %s on %s -- %s",
                     config.severity.upper(), config.type, config.component, expected_impact)

    def increment_steps(self) -> None:
        self.step_counter += 1

    def reset_run_state(self) -> None:
        """
        Clear per-attempt state before a task attempt begins.

        Retries re-execute the agent, so without this the same fault is logged
        once per attempt and inflates the injection counts that the regression
        catch rate is computed from. Reports must describe the attempt whose
        result was kept, not the sum of every attempt made.

        The random stream is deliberately not reset: a retry is a fresh draw, and
        rewinding it would make a probabilistic fault repeat forever once it
        happened to fire.
        """
        self.logs.clear()
        self.step_counter = 0

    def save_reports(self, workspace_path: str = ".") -> None:
        """
        Generates fault_log.json and fault_report.json inside the workspace,
        aggregating this engine's logs with those of every forked child.
        """
        log_file = os.path.join(workspace_path, "fault_log.json")
        report_file = os.path.join(workspace_path, "fault_report.json")

        entries = self.collect_logs()
        serialized_logs = [log.model_dump() for log in entries]

        # 1. Save fault_log.json
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(serialized_logs, f, indent=2)

        # 2. Summarize and save fault_report.json
        severity_counts: Dict[str, int] = {}
        component_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        for entry in entries:
            severity_counts[entry.severity] = severity_counts.get(entry.severity, 0) + 1
            component_counts[entry.component] = component_counts.get(entry.component, 0) + 1
            type_counts[entry.type] = type_counts.get(entry.type, 0) + 1

        report = {
            "summary": {
                "total_faults_injected": len(entries),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "seed": self.seed,
                "severity_distribution": severity_counts,
                "component_distribution": component_counts,
                "type_distribution": type_counts
            },
            "injections": serialized_logs
        }

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info("Fault logs written to %s and %s", log_file, report_file)
