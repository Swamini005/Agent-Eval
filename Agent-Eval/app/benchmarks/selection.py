"""
Capability-based task selection: run the tasks an agent can actually attempt.

Running the whole suite against every agent wastes time and, worse, reports
misleading numbers. A task that needs `get_policy_document` cannot be attempted
by an agent that has no such tool -- it fails, but the failure says nothing about
the agent's quality, and it drags the headline score down.

This is the same rule already applied to regressions. The experiment reports
`inert_regressions` for faults an agent cannot exhibit rather than counting them
as misses; tasks the agent cannot attempt are reported as skipped rather than
counted as failures.

A task's requirements come from what it already declares -- `expected_tools` and
`ground_truth.must_call_tools` -- so nothing new has to be written per task.
Skipped tasks are always listed with a reason: silently shrinking the suite would
be far worse than running too much of it.
"""

from typing import Any, Dict, Iterable, List, Set, Tuple

from app.benchmarks.models import UnifiedBenchmarkTask


class AgentCapabilities:
    """What an agent can do, as declared by its adapter."""

    def __init__(self, tools: Iterable[str] = None, plans: bool = False,
                 retrieves: bool = False):
        # None means the tool surface could not be determined -- a third-party
        # agent, typically. Every task is then attempted: under-running a
        # supplied agent yields a fast, flattering, meaningless result.
        self.tools_known = tools is not None
        self.tools: Set[str] = set(tools or ())
        self.plans = plans
        self.retrieves = retrieves

    def __repr__(self) -> str:
        if not self.tools_known:
            return "AgentCapabilities(tools=unknown, all tasks attempted)"
        return (f"AgentCapabilities(tools={len(self.tools)}, "
                f"plans={self.plans}, retrieves={self.retrieves})")


def required_tools(task: UnifiedBenchmarkTask) -> Set[str]:
    """
    Tools a task cannot be attempted without.

    Drawn from what the task already declares. `must_not_call_tools` is excluded
    deliberately: a task asserting the agent does *not* call something is still
    meaningful, and arguably more so, when the agent lacks that tool entirely.
    """
    ground_truth = task.ground_truth or {}
    needed = set(task.expected_tools or [])
    needed |= set(ground_truth.get("must_call_tools", []))
    needed |= set(ground_truth.get("must_call_tools_in_order", []))

    trajectory = ground_truth.get("trajectory_match") or {}
    for entry in trajectory.get("reference_tools", []):
        needed.add(entry if isinstance(entry, str) else entry.get("name", ""))

    return {tool for tool in needed if tool}


def skip_reason(task: UnifiedBenchmarkTask,
                capabilities: AgentCapabilities) -> str:
    """Why this agent cannot attempt this task, or "" when it can."""
    if not capabilities.tools_known:
        return ""

    missing = required_tools(task) - capabilities.tools
    if missing:
        return f"agent has no {', '.join(sorted(missing))}"

    # A plan-dependent task cannot be evaluated against an agent with no planner.
    # The ReAct target is exactly this case, and it is why planner faults measure
    # zero against it.
    if task.ground_truth and task.ground_truth.get("requires_planner") and not capabilities.plans:
        return "agent has no planner"

    return ""


def select(tasks: List[UnifiedBenchmarkTask],
           capabilities: AgentCapabilities) -> Tuple[List[UnifiedBenchmarkTask], List[Dict[str, Any]]]:
    """
    Split a suite into what this agent can attempt and what it cannot.

    Returns (applicable, skipped), where each skipped entry carries the task id
    and the reason, so a report can state what was not run and why.
    """
    applicable, skipped = [], []
    for task in tasks:
        reason = skip_reason(task, capabilities)
        if reason:
            skipped.append({
                "task_id": task.id,
                "category": task.category,
                "reason": reason,
                "required_tools": sorted(required_tools(task)),
            })
        else:
            applicable.append(task)
    return applicable, skipped


def describe(selected: List[UnifiedBenchmarkTask],
             skipped: List[Dict[str, Any]]) -> str:
    """One-line summary for a CLI."""
    total = len(selected) + len(skipped)
    if not skipped:
        return f"{total} tasks, all applicable to this agent"
    return (f"{len(selected)} of {total} tasks applicable; "
            f"{len(skipped)} skipped (agent lacks the required tools)")
