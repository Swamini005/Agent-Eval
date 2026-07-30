from typing import Dict, Any, List, Callable
from app.evaluation.metrics.base import BaseMetricPlugin
from app.evaluation.metrics.registry import MetricRegistry
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput, MetricResult
from app.packs import PackRegistry


@MetricRegistry.register("assertions")
class AssertionMetric(BaseMetricPlugin):
    """
    Evaluates a task's declared ground-truth assertions against what the agent did.

    Every other metric produces a similarity score -- useful for tracking drift,
    but never a definitive verdict. This one answers the question a CI gate
    actually needs: did the agent do the specific things this task requires?

    Assertions are declared as data on the task, so adding coverage means editing
    a task file rather than writing Python. Supported clauses:

        must_call_tools:            [str]  every listed tool was called
        must_not_call_tools:        [str]  none of the listed tools were called
        must_call_tools_in_order:   [str]  listed tools appear in this relative order
        must_not_call_destructive:  bool   no tool the domain pack marks destructive ran
        response_must_contain:      [str]  every string appears (case-insensitive)
        response_must_not_contain:  [str]  no string appears (case-insensitive)
        max_tool_calls:             int    at most this many tool calls were made
        max_latency_seconds:        float  completed within this budget

    The score is the fraction of declared assertions that held. `all_passed` in
    the details is the binary verdict a gate should read; the fraction exists so
    a partial regression is visible rather than collapsing to zero.
    """

    def evaluate(
        self,
        task: EvaluationTaskInput,
        execution: EvaluationExecutionInput,
        fault_report: Dict[str, Any]
    ) -> MetricResult:
        ground_truth = task.ground_truth or {}
        if not ground_truth:
            # No assertions declared. Scoring this 1.0 would reward tasks for
            # having no checks, and 0.0 would punish them; neither is a finding.
            return MetricResult(
                metric_name="assertions",
                score=0.0,
                measured=False,
                details={"reason": "No ground_truth assertions declared for this task."}
            )

        called = [c.get("tool_name") for c in execution.tool_calls if c.get("tool_name")]
        response = (execution.response or "").lower()
        checks: List[Dict[str, Any]] = []

        def record(name: str, passed: bool, detail: str) -> None:
            checks.append({"assertion": name, "passed": passed, "detail": detail})

        handlers: Dict[str, Callable[[Any], None]] = {
            "must_call_tools": lambda v: [
                record("must_call_tools", tool in called, f"{tool}: {'called' if tool in called else 'never called'}")
                for tool in v
            ],
            "must_not_call_tools": lambda v: [
                record("must_not_call_tools", tool not in called, f"{tool}: {'called' if tool in called else 'not called'}")
                for tool in v
            ],
            "must_call_tools_in_order": lambda v: record(
                "must_call_tools_in_order",
                self._is_subsequence(v, called),
                f"expected order {v}, observed {called}"
            ),
            "must_not_call_destructive": lambda v: self._check_destructive(
                v, task, called, record
            ),
            "response_must_contain": lambda v: [
                record("response_must_contain", text.lower() in response, f"{text!r}: {'present' if text.lower() in response else 'absent'}")
                for text in v
            ],
            "response_must_not_contain": lambda v: [
                record("response_must_not_contain", text.lower() not in response, f"{text!r}: {'present' if text.lower() in response else 'absent'}")
                for text in v
            ],
            "max_tool_calls": lambda v: record(
                "max_tool_calls", len(called) <= v, f"{len(called)} calls, limit {v}"
            ),
            "max_latency_seconds": lambda v: record(
                "max_latency_seconds",
                execution.latency_seconds <= v,
                f"{execution.latency_seconds:.3f}s, limit {v}s"
            ),
            "trajectory_match": lambda v: self._check_trajectory(v, execution.tool_calls, record),
        }

        unknown = [clause for clause in ground_truth if clause not in handlers]
        for clause, value in ground_truth.items():
            if clause in handlers:
                handlers[clause](value)

        if not checks:
            return MetricResult(
                metric_name="assertions",
                score=0.0,
                measured=False,
                details={
                    "reason": "ground_truth declared no recognised assertions.",
                    "unknown_clauses": unknown,
                    "supported_clauses": sorted(handlers)
                }
            )

        passed = [c for c in checks if c["passed"]]
        failed = [c for c in checks if not c["passed"]]

        return MetricResult(
            metric_name="assertions",
            score=round(len(passed) / len(checks), 3),
            details={
                "all_passed": not failed,
                "passed": len(passed),
                "total": len(checks),
                "failures": failed,
                "checks": checks,
                # Surfaced rather than ignored: a typo in a task file would
                # otherwise silently reduce that task's coverage to nothing.
                "unknown_clauses": unknown
            }
        )

    @staticmethod
    def _check_trajectory(
        spec: Dict[str, Any],
        tool_calls: List[Dict[str, Any]],
        record: Callable[[str, bool, str], None]
    ) -> None:
        """
        Compare the observed tool-call trajectory against a reference.

        Delegated to `agentevals`. A failure to evaluate is recorded as a failed
        assertion rather than swallowed: a task that declared a trajectory
        requirement must not pass merely because the check could not run.
        """
        from app.evaluation.trajectory import match

        try:
            result = match(
                tool_calls=tool_calls,
                reference_tools=spec.get("reference_tools", []),
                mode=spec.get("mode", "superset"),
                tool_args_match_mode=spec.get("tool_args_match_mode", "ignore"),
            )
            record("trajectory_match", result["passed"], result["detail"])
        except Exception as e:
            record("trajectory_match", False, f"could not evaluate: {e}")

    @staticmethod
    def _is_subsequence(expected: List[str], observed: List[str]) -> bool:
        """True when `expected` appears in `observed` in order, gaps allowed."""
        it = iter(observed)
        return all(tool in it for tool in expected)

    @staticmethod
    def _check_destructive(
        enabled: bool,
        task: EvaluationTaskInput,
        called: List[str],
        record: Callable[[str, bool, str], None]
    ) -> None:
        """Assert no destructive tool ran, using the task's domain pack."""
        if not enabled:
            return

        pack = PackRegistry.get(getattr(task, "domain", ""))
        if pack is None:
            record(
                "must_not_call_destructive", False,
                f"cannot evaluate: no domain pack for '{getattr(task, 'domain', '')}'"
            )
            return

        offenders = [tool for tool in called if tool in pack.destructive_tools]
        record(
            "must_not_call_destructive",
            not offenders,
            f"destructive tools called: {offenders}" if offenders else "none called"
        )
