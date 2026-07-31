"""
Impact analysis: run the tasks a commit could actually have affected.

Evaluating the whole suite on every commit is slow and mostly wasted -- a change
to one tool cannot alter the outcome of tasks that never call it. This maps
changed files to the tasks whose result could plausibly move, so a typical pull
request evaluates a handful of tasks instead of thirty.

The governing rule is that **unknown means everything**. A path this module does
not recognise selects the full suite, because under-selecting silently is the one
failure that matters here: it produces a fast green result that proves nothing,
which is exactly the class of bug this project exists to catch. Over-selecting
only costs time.
"""

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple

from app.benchmarks.models import UnifiedBenchmarkTask
from app.benchmarks.selection import required_tools

# Paths that change how any task is executed or scored. A change here invalidates
# every previous result, so the whole suite runs.
GLOBAL_IMPACT_PREFIXES = (
    "app/agent/",          # the agent under test
    "app/adapters/",       # how it is driven
    "app/evaluation/",     # how it is scored
    "app/faults/",         # what is injected
    "app/benchmarks/",     # how it is run and selected
    "app/packs.py",
    "app/pricing.py",
    "app/budget.py",
    "app/config.py",
    "packs/",              # domain vocabulary
    "agents/",             # agent variants
    "regressions.yaml",
    "pricing.yaml",
    "requirements.txt",
)

# Build output and caches. Matched anywhere in the path rather than as a prefix,
# because these sit beside the source they are generated from -- and a stale
# app/adapters/__pycache__/base.cpython-312.pyc read as a change to the adapter
# contract, which escalated the run to "affects every task" and silently undid
# triage.
ARTIFACT_MARKERS = ("__pycache__/", ".pyc", ".pyo", ".pytest_cache/", "reports/", ".venv/")


def is_artifact(path: str) -> bool:
    return any(marker in path for marker in ARTIFACT_MARKERS)


# Paths that cannot change any task's outcome.
NO_IMPACT_PREFIXES = (
    "tests/",
    "docs/",
    "scripts/",
    ".github/",
    "README.md",
    ".gitignore",
    ".streamlit/",
    "dashboard.py",
)


def changed_files(base_ref: str = "origin/main") -> List[str]:
    """
    Files changed against a base ref, as forward-slash repo-relative paths.

    Returns [] when git cannot answer -- a shallow clone, a missing ref, no git
    at all. Callers treat that as "cannot determine impact" and run everything.
    """
    for args in (["diff", "--name-only", f"{base_ref}...HEAD"],
                 ["diff", "--name-only", base_ref]):
        try:
            out = subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.DEVNULL, timeout=30
            )
        except Exception:
            continue
        files = [line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()]
        if files:
            return files
    return []


def changed_tool_names(paths: List[str], base_ref: str) -> Set[str]:
    """
    Tool functions mentioned in the diff of the agent's tool module.

    Lets a change to one tool select only the tasks that call it, rather than the
    whole suite. Returns empty when the diff cannot be read, which the caller
    treats as "all tools affected".
    """
    tool_files = [p for p in paths if p.endswith("agent/tools.py")]
    if not tool_files:
        return set()

    try:
        diff = subprocess.check_output(
            ["git", "diff", f"{base_ref}...HEAD", "--", *tool_files],
            text=True, stderr=subprocess.DEVNULL, timeout=30,
        )
    except Exception:
        return set()

    from app.agent.tools import tools_map
    changed = set()
    for line in diff.splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        for name in tools_map:
            if name in line:
                changed.add(name)
    return changed


def changed_task_ids(paths: List[str], base_ref: str) -> Set[str]:
    """Task ids added or modified in any tasks/*.json touched by the diff."""
    task_files = [p for p in paths if p.startswith("tasks/") and p.endswith(".json")]
    changed: Set[str] = set()

    for path in task_files:
        try:
            before = subprocess.check_output(
                ["git", "show", f"{base_ref}:{path}"],
                text=True, stderr=subprocess.DEVNULL, timeout=30,
            )
            previous = {t["id"]: t for t in json.loads(before)}
        except Exception:
            previous = {}

        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            current = {t["id"]: t for t in json.load(f)}

        for task_id, task in current.items():
            if previous.get(task_id) != task:
                changed.add(task_id)

    return changed


def select_affected(
    tasks: List[UnifiedBenchmarkTask],
    base_ref: str = "origin/main",
    paths: Optional[List[str]] = None,
) -> Tuple[List[UnifiedBenchmarkTask], Dict[str, Any]]:
    """
    Tasks a commit could have affected, with the reasoning behind the choice.

    Returns (selected, rationale). The rationale is always reported so a short
    run can be seen to be short for a defensible reason rather than by accident.
    """
    paths = changed_files(base_ref) if paths is None else paths

    if not paths:
        return list(tasks), {
            "scope": "all",
            "reason": f"could not determine changes against {base_ref}; running the full suite",
            "changed_files": [],
        }

    relevant = [p for p in paths
                if not p.startswith(NO_IMPACT_PREFIXES) and not is_artifact(p)]
    if not relevant:
        return [], {
            "scope": "none",
            "reason": "only documentation, tests, CI configuration or build "
                      "artifacts changed",
            "changed_files": paths,
        }

    global_hits = [p for p in relevant if p.startswith(GLOBAL_IMPACT_PREFIXES)
                   and not p.endswith("agent/tools.py")]
    if global_hits:
        return list(tasks), {
            "scope": "all",
            "reason": f"changes to {', '.join(sorted(set(global_hits))[:3])} affect every task",
            "changed_files": paths,
        }

    selected: Set[str] = set()
    reasons = []

    tool_names = changed_tool_names(relevant, base_ref)
    if any(p.endswith("agent/tools.py") for p in relevant):
        if not tool_names:
            return list(tasks), {
                "scope": "all",
                "reason": "agent/tools.py changed but the affected tools could not be identified",
                "changed_files": paths,
            }
        for task in tasks:
            if required_tools(task) & tool_names:
                selected.add(task.id)
        reasons.append(f"tools changed: {', '.join(sorted(tool_names))}")

    task_ids = changed_task_ids(relevant, base_ref)
    if task_ids:
        selected |= task_ids
        reasons.append(f"{len(task_ids)} task definition(s) changed")

    unexplained = [
        p for p in relevant
        if not p.endswith("agent/tools.py") and not p.startswith("tasks/")
    ]
    if unexplained:
        return list(tasks), {
            "scope": "all",
            "reason": f"unrecognised path {unexplained[0]}; running everything rather than guessing",
            "changed_files": paths,
        }

    chosen = [t for t in tasks if t.id in selected]
    return chosen, {
        "scope": "affected",
        "reason": "; ".join(reasons) or "no task-affecting changes",
        "changed_files": paths,
        "selected_task_ids": sorted(selected),
    }
