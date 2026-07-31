"""
Run one ad-hoc task and print its verdict.

    python run_task.py path/to/task.json [--target=react] [--json]

For evaluating a task the suite has never seen -- a reviewer's own case, or a
scenario being drafted before it is added to tasks/. Prints every assertion that
was checked and why it held or failed, then exits 0 or 1.

The file may contain a single task object or a list of them, in the same shape
as tasks/dev.json. Any task lacking `ground_truth` is rejected rather than run:
a prompt with no checker produces an opinion, not a verdict.
"""

import json
import sys
import tempfile

import app.adapters  # noqa: F401  (registers the concrete adapters)
from app.adapters.factory import AgentFactory
from app.benchmarks.models import UnifiedBenchmarkTask
from app.logging_setup import configure, verbosity_from_args
from app.benchmarks.runner import BenchmarkRunner
from app.evaluation.engine import EvaluationEngine
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput

DEFAULT_TARGET = "react"

# Applied to any field the caller omits, so a reviewer can hand over the three
# fields that matter -- prompt, domain, ground_truth -- without boilerplate.
#
# `domain` is deliberately absent: defaulting it would score a task from an
# unknown domain using another domain's vocabulary. An undeclared domain has no
# pack, so domain-dependent checks report themselves unmeasured instead.
DEFAULTS = {
    "benchmark": "adhoc",
    "category": "general",
    "domain": "",
    "difficulty": "medium",
    "expected_answer": None,
    "expected_tools": [],
    "metadata": {},
}


def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        payload = [payload]

    tasks = []
    for index, raw in enumerate(payload):
        if not raw.get("ground_truth"):
            raise ValueError(
                f"Task {raw.get('id', index)} declares no ground_truth. "
                f"Without assertions there is nothing to verify."
            )
        tasks.append(UnifiedBenchmarkTask(**{**DEFAULTS, "id": f"adhoc-{index}", **raw}))
    return tasks


def main():
    configure(verbosity_from_args(sys.argv))
    args = [a for a in sys.argv[1:]]
    paths = [a for a in args if not a.startswith("--")]
    target = next((a.split("=", 1)[1] for a in args if a.startswith("--target=")), DEFAULT_TARGET)
    as_json = "--json" in args

    if len(paths) != 1:
        # Silently ignoring extra paths would run a subset of what was asked for
        # and report a verdict for the whole set.
        print(__doc__)
        if len(paths) > 1:
            print(f"ERROR: expected one task file, got {len(paths)}: {paths}")
        return 2

    tasks = load_tasks(paths[0])

    runner = BenchmarkRunner(
        lambda task_id: AgentFactory.create_agent(target),
        concurrency=min(4, len(tasks)),
        max_retries=0,
    )

    with tempfile.TemporaryDirectory() as workdir:
        execution = runner.run_benchmark(tasks, output_dir=workdir)
        by_id = {r["task_id"]: r for r in execution["tasks"]}

        reports = EvaluationEngine().evaluate_run(
            tasks=[
                EvaluationTaskInput(
                    task_id=t.id, benchmark=t.benchmark, category=t.category, difficulty=t.difficulty,
                    domain=t.domain, prompt=t.prompt, expected_answer=t.expected_answer,
                    expected_tools=t.expected_tools, ground_truth=t.ground_truth,
                ) for t in tasks
            ],
            executions=[
                EvaluationExecutionInput(
                    task_id=r["task_id"], category=r.get("category", "general"),
                    response=r["response"], latency_seconds=r["latency_seconds"],
                    cost_usd=r["cost_usd"], tool_calls=r["tool_calls"], tokens=r["tokens"],
                    memory_state=r["memory_state"], retrieval_documents=r["retrieval_documents"],
                    reasoning_nodes=r["reasoning_nodes"], errors=r["errors"],
                ) for r in execution["tasks"]
            ],
            fault_report={"injections": []},
            output_dir=workdir,
        )

    if as_json:
        print(json.dumps(reports["results"], indent=2))
        return 0 if all_passed(reports) else 1

    print("=" * 70)
    print(f"  AD-HOC TASK RUN   target={target}   tasks={len(tasks)}")
    print("=" * 70)

    for result in reports["results"]:
        record = by_id.get(result["task_id"], {})
        assertions = result["details"].get("assertions", {})
        passed = assertions.get("all_passed")

        print(f"\n[{'PASS' if passed else 'FAIL'}] {result['task_id']}")
        print(f"  prompt   : {result['prompt']}")
        print(f"  response : {(record.get('response') or '')[:160]}")
        print(f"  tools    : {[c.get('tool_name') for c in record.get('tool_calls', [])]}")
        print(f"  latency  : {record.get('latency_seconds')}s")

        # Surfaced before the assertions: when a task errored, the assertions
        # are downstream noise and the error is the finding.
        for error in record.get("errors", []):
            print(f"    ERROR {error[:220]}")

        for check in assertions.get("checks", []):
            mark = "ok  " if check["passed"] else "FAIL"
            print(f"    {mark} {check['assertion']}: {check['detail']}")

        unknown = assertions.get("unknown_clauses") or []
        if unknown:
            print(f"    !! unrecognised clauses ignored: {unknown}")

    ok = all_passed(reports)
    print("\n" + "=" * 70)
    print("VERDICT: " + ("all tasks passed" if ok else "at least one task failed"))
    print("=" * 70)
    return 0 if ok else 1


def all_passed(reports) -> bool:
    return all(
        r["details"].get("assertions", {}).get("all_passed")
        for r in reports["results"]
    )


if __name__ == "__main__":
    sys.exit(main())
