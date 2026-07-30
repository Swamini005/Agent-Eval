"""
Task suites loaded from data, with a content hash proving what was evaluated.

Tasks defined as Python literals inside a pipeline script cannot be audited: the
suite and the tuning that follows it live in the same file, and there is no way
to show a reviewer that the test set predates the improvements measured against
it. Suites live in `tasks/` as JSON and carry a hash, so a report can name the
exact task set that produced it.

`dev` is the working set. `holdout` is sealed -- it must not be inspected while
tuning, and is run once at the end to detect overfitting to the dev set.
"""

import hashlib
import json
import os
from typing import Dict, List, Any

from app.benchmarks.models import UnifiedBenchmarkTask
from app.benchmarks.registry import BenchmarkRegistry
import app.benchmarks.providers.harbor          # noqa: F401  (registers provider)
import app.benchmarks.providers.context_bench   # noqa: F401
import app.benchmarks.providers.t3_bench        # noqa: F401
import app.benchmarks.providers.custom_json     # noqa: F401

TASKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "tasks")


class TaskSuite:
    """A named, hashed collection of benchmark tasks."""

    def __init__(self, name: str, tasks: List[UnifiedBenchmarkTask], sha: str):
        self.name = name
        self.tasks = tasks
        self.sha = sha

    def __len__(self) -> int:
        return len(self.tasks)

    def categories(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for task in self.tasks:
            counts[task.category] = counts.get(task.category, 0) + 1
        return dict(sorted(counts.items()))


def suite_path(name: str) -> str:
    return os.path.join(TASKS_DIR, f"{name}.json")


def available_suites() -> List[str]:
    if not os.path.isdir(TASKS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(TASKS_DIR) if f.endswith(".json"))


def compute_sha(raw: List[Dict[str, Any]]) -> str:
    """
    Content hash of a task list.

    Computed over canonical JSON -- sorted keys, no incidental whitespace -- so
    reformatting the file does not change the hash but editing a prompt, an
    expected tool or an assertion does.
    """
    canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _is_unified(raw: List[Dict[str, Any]]) -> bool:
    """True when the file is already in UnifiedBenchmarkTask shape."""
    return bool(raw) and "id" in raw[0] and "prompt" in raw[0]


def load_suite(name: str, provider: str = None) -> TaskSuite:
    """
    Load a suite by name from tasks/<name>.json.

    Files already in UnifiedBenchmarkTask shape load directly. Anything else is
    treated as a foreign benchmark export and normalised through a registered
    provider, so a Harbor or ContextBench dump can be dropped into tasks/ and
    evaluated without being rewritten by hand. The provider is taken from the
    `provider` argument, or inferred from the file's own `benchmark` field.
    """
    path = suite_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No task suite '{name}' at {path}. Available: {available_suites()}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not _is_unified(raw):
        provider_name = provider or (raw[0].get("benchmark") if raw else None)
        if not provider_name:
            raise ValueError(
                f"Suite '{name}' is not in unified format and declares no benchmark; "
                f"pass provider= explicitly. Registered: {BenchmarkRegistry.available()}"
            )
        normalized = BenchmarkRegistry.get_provider(provider_name).load_tasks(raw)
        return TaskSuite(name=name, tasks=normalized, sha=compute_sha(raw))

    ids = [task["id"] for task in raw]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Suite '{name}' has duplicate task ids: {sorted(duplicates)}")

    return TaskSuite(
        name=name,
        tasks=[UnifiedBenchmarkTask(**task) for task in raw],
        sha=compute_sha(raw),
    )
