"""
DeepEval interop: expose this project's metrics through a standard interface.

DeepEval is the de-facto pytest-native harness for LLM evaluation, so wiring the
suite into it means `deepeval test run` and any DeepEval-aware tooling works
against these tasks without learning a bespoke format.

What is deliberately *not* adopted is DeepEval's LLM-judged metrics. Every metric
here is deterministic Python reading observed state and tool calls, so a run
costs no judge tokens and returns the same verdict every time. A judge in the
gating path would put model noise between a code change and its verdict, which
is the opposite of what a merge blocker is for.

The bridge is therefore thin by design: DeepEval supplies the runner, the report
and the CI exit code; this project supplies the measurements.
"""

from typing import Any, Dict, List, Optional

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, ToolCall

from app.evaluation.metrics.base import BaseMetricPlugin
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput

# Keys under LLMTestCase.metadata carrying the full-fidelity objects. The
# standard LLMTestCase fields are populated too, so DeepEval's own reporting
# still works, but they are lossy: they have nowhere to put memory state,
# reasoning nodes or the fault report a metric may need.
TASK_KEY = "agenteval_task"
EXECUTION_KEY = "agenteval_execution"
FAULT_REPORT_KEY = "agenteval_fault_report"


def to_test_case(
    task: EvaluationTaskInput,
    execution: EvaluationExecutionInput,
    fault_report: Optional[Dict[str, Any]] = None,
) -> LLMTestCase:
    """Build a DeepEval test case that is both well-formed and lossless."""
    return LLMTestCase(
        name=task.task_id,
        input=task.prompt,
        actual_output=execution.response,
        expected_output=task.expected_answer,
        retrieval_context=[
            str(doc.get("content", "")) for doc in execution.retrieval_documents
        ] or None,
        tools_called=[
            ToolCall(name=call.get("tool_name", "unknown"),
                     input_parameters=call.get("args") or {},
                     output=str(call.get("result", "")))
            for call in execution.tool_calls
        ] or None,
        expected_tools=[ToolCall(name=name) for name in task.expected_tools] or None,
        token_cost=execution.cost_usd,
        completion_time=execution.latency_seconds,
        tags=[task.category, task.domain],
        metadata={
            TASK_KEY: task,
            EXECUTION_KEY: execution,
            FAULT_REPORT_KEY: fault_report or {"injections": []},
        },
    )


class AgentEvalMetric(BaseMetric):
    """
    Adapts one BaseMetricPlugin to DeepEval's metric interface.

    Success is not always `score >= threshold`. Assertion results carry an
    explicit all-or-nothing verdict, and using a fraction there would let a task
    that failed a safety assertion pass on the strength of the checks it did
    satisfy.
    """

    def __init__(self, plugin: BaseMetricPlugin, threshold: float = 0.5):
        self.plugin = plugin
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True
        self.strict_mode = False
        self.evaluation_model = "deterministic (no LLM judge)"
        self.score = 0.0
        self.success = False
        self.reason = ""
        self.skipped = False

    @property
    def __name__(self):
        return self.plugin.name

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        metadata = test_case.metadata or {}
        task = metadata.get(TASK_KEY)
        execution = metadata.get(EXECUTION_KEY)
        if task is None or execution is None:
            raise ValueError(
                f"Test case '{test_case.name}' was not built by to_test_case(); "
                f"metadata is missing the task and execution records this metric needs."
            )

        result = self.plugin.evaluate(task, execution, metadata.get(FAULT_REPORT_KEY, {}))
        self.score = result.score

        if not result.measured:
            # Nothing to evaluate. Reported as skipped rather than passed or
            # failed, because both would misdescribe the agent.
            self.skipped = True
            self.success = True
            self.reason = f"not measured: {result.details.get('reason', 'no applicable evidence')}"
            return self.score

        details = result.details
        if "all_passed" in details:
            self.success = bool(details["all_passed"])
            failures = details.get("failures", [])
            self.reason = (
                f"all {details.get('total', 0)} assertions held"
                if self.success else
                "failed: " + "; ".join(
                    f"{f['assertion']} ({f['detail']})" for f in failures
                )
            )
        else:
            self.success = self.score >= self.threshold
            self.reason = details.get("reason") or f"score {self.score:.3f} vs threshold {self.threshold:.2f}"

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


def build_metrics(names: Optional[List[str]] = None, threshold: float = 0.5) -> List[AgentEvalMetric]:
    """
    Wrap the registered metric plugins as DeepEval metrics.

    Defaults to the whole registry, so a metric added to this project appears in
    DeepEval runs without touching this module.
    """
    from app.evaluation.metrics.registry import MetricRegistry
    from app.evaluation.engine import EvaluationEngine

    # Constructing the engine imports every metric module, which is what
    # populates the registry via their decorators.
    EvaluationEngine()
    plugins = MetricRegistry.get_all_metrics()
    if names:
        wanted = set(names)
        plugins = [p for p in plugins if p.name in wanted]
    return [AgentEvalMetric(plugin, threshold=threshold) for plugin in plugins]
