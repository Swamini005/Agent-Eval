"""
Regression experiments: measuring the instrument, not just the agent.

The catch rate this replaces was computed from a single faulted run by asking
whether each task scored below a fixed threshold. With a baseline already below
that threshold, almost every task qualified, so the headline "100% detection"
was close to tautological -- it never compared the faulted run against anything.

This module runs a *paired* experiment instead. A clean arm establishes what each
task does with no faults; one arm per planted regression re-runs the identical
suite with a single fault enabled. A regression counts as detected only when the
faulted arm's pass rate drops below the clean arm's by more than the noise the
clean arm itself exhibits.

Running several regressions of decreasing severity also yields the suite's
minimum detectable effect: the smallest true degradation it can resolve. That
number is what makes a detection claim defensible, because it states the
conditions under which the claim holds.
"""

import hashlib
import json
import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

from app.adapters.base import BaseAgentAdapter
from app.benchmarks.models import UnifiedBenchmarkTask
from app.benchmarks.runner import BenchmarkRunner
from app.evaluation import statistics as stats
from app.evaluation.engine import EvaluationEngine
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput
from app.faults.engine import FaultInjectionEngine
from app.faults.middleware import FaultInjectionMiddleware
from app.faults.models import FaultConfig

AgentBuilder = Callable[[], BaseAgentAdapter]


class Arm:
    """One experimental condition: a named fault set applied to the whole suite."""

    def __init__(self, label: str, faults: List[FaultConfig], effect_label: str = ""):
        self.label = label
        self.faults = faults
        self.effect_label = effect_label

    @property
    def is_clean(self) -> bool:
        return not self.faults


class RegressionExperiment:
    """
    Runs a suite across several arms and seeds, then derives detection metrics.

    Significance level for per-task detection. Fixed rather than tunable: a
    threshold that can be relaxed per run is not evidence.

    A task's verdict is its assertion outcome -- every assertion held, or one did
    not. That is a genuine binary rather than a score compared against a magic
    threshold, which is what allows pass rates and confidence intervals to mean
    something.
    """

    ALPHA = 0.05

    def __init__(
        self,
        tasks: List[UnifiedBenchmarkTask],
        agent_builder: AgentBuilder,
        seeds: List[int],
        concurrency: int = 4,
        target_label: str = "unknown",
        suite_sha: str = "",
        cache_dir: Optional[str] = None,
        replay: bool = False,
    ):
        """
        Args:
            target_label: Names the agent under test. Part of the cache key, so
                results from one target are never served for another.
            suite_sha: Content hash of the task set, also part of the cache key:
                editing a prompt or an assertion must invalidate cached verdicts.
            cache_dir: Where per-(arm, seed) verdicts are stored. The experiment
                is (arms + 1) x seeds full suite runs, so re-running it after a
                reporting change would otherwise repeat every execution.
            replay: Serve only from cache and fail on a miss. Lets a demo render
                without executing the agent, and without a network.
        """
        self.tasks = tasks
        self.agent_builder = agent_builder
        self.seeds = seeds
        self.concurrency = concurrency
        self.target_label = target_label
        self.suite_sha = suite_sha
        self.cache_dir = cache_dir
        self.replay = replay
        self.cache_hits = 0
        self.cache_misses = 0

    # -- caching -----------------------------------------------------------

    def _cache_key(self, arm: Arm, seed: int) -> str:
        """
        Identity of one (arm, seed) execution.

        Covers everything that can change a verdict: which agent, which tasks,
        which faults with which parameters, and which seed. Fault configs are
        serialised in full rather than by id, so editing a delay or a probability
        invalidates the entry instead of silently reusing it.
        """
        payload = json.dumps({
            "target": self.target_label,
            "suite_sha": self.suite_sha,
            "task_ids": sorted(t.id for t in self.tasks),
            "arm": arm.label,
            "faults": sorted(f.model_dump_json() for f in arm.faults),
            "seed": seed,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def _cache_path(self, arm: Arm, seed: int) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{self._cache_key(arm, seed)}.json")

    def _read_cache(self, arm: Arm, seed: int) -> Optional[Dict[str, bool]]:
        path = self._cache_path(arm, seed)
        if not path or not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)["verdicts"]

    def _write_cache(self, arm: Arm, seed: int, verdicts: Dict[str, bool]) -> None:
        path = self._cache_path(arm, seed)
        if not path:
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "target": self.target_label,
                "suite_sha": self.suite_sha,
                "arm": arm.label,
                "seed": seed,
                "verdicts": verdicts,
            }, f, indent=2)

    # -- execution ---------------------------------------------------------

    def _run_once(self, arm: Arm, seed: int, workdir: str) -> Dict[str, bool]:
        """Execute the suite once and return {task_id: all assertions held}."""
        fault_engine = FaultInjectionEngine(arm.faults, seed=seed)
        domains = {t.id: t.domain for t in self.tasks}

        def build(task_id: str) -> BaseAgentAdapter:
            return FaultInjectionMiddleware(
                self.agent_builder(),
                fault_engine.fork(task_id),
                domain=domains.get(task_id, ""),
            )

        runner = BenchmarkRunner(build, concurrency=self.concurrency, max_retries=0)
        execution = runner.run_benchmark(self.tasks, output_dir=workdir)

        fault_engine.save_reports(workspace_path=workdir)
        with open(os.path.join(workdir, "fault_report.json"), "r", encoding="utf-8") as f:
            fault_report = json.load(f)

        reports = EvaluationEngine().evaluate_run(
            tasks=[
                EvaluationTaskInput(
                    task_id=t.id, benchmark=t.benchmark, category=t.category,
                    domain=t.domain, prompt=t.prompt, expected_answer=t.expected_answer,
                    expected_tools=t.expected_tools, ground_truth=t.ground_truth,
                ) for t in self.tasks
            ],
            executions=[
                EvaluationExecutionInput(
                    task_id=e["task_id"], category=e.get("category", "general"),
                    response=e["response"], latency_seconds=e["latency_seconds"],
                    cost_usd=e["cost_usd"], tool_calls=e["tool_calls"], tokens=e["tokens"],
                    memory_state=e["memory_state"], retrieval_documents=e["retrieval_documents"],
                    reasoning_nodes=e["reasoning_nodes"], errors=e["errors"],
                ) for e in execution["tasks"]
            ],
            fault_report=fault_report,
            output_dir=workdir,
            run_metadata={"arm": arm.label, "seed": seed},
        )

        verdicts = {}
        for result in reports["results"]:
            assertions = result.get("details", {}).get("assertions", {})
            verdicts[result["task_id"]] = bool(assertions.get("all_passed"))
        return verdicts

    def run_arm(self, arm: Arm) -> Dict[str, Any]:
        """Run every seed for one arm and aggregate per-task pass rates."""
        per_seed: List[Dict[str, bool]] = []
        for seed in self.seeds:
            cached = self._read_cache(arm, seed)
            if cached is not None:
                self.cache_hits += 1
                per_seed.append(cached)
                continue

            if self.replay:
                raise RuntimeError(
                    f"--replay was requested but no cached result exists for "
                    f"arm '{arm.label}' seed {seed} "
                    f"(target={self.target_label}, suite_sha={self.suite_sha}). "
                    f"Run the experiment once without --replay to populate the cache."
                )

            self.cache_misses += 1
            with tempfile.TemporaryDirectory() as workdir:
                verdicts = self._run_once(arm, seed, workdir)
            self._write_cache(arm, seed, verdicts)
            per_seed.append(verdicts)

        per_task = {}
        for task in self.tasks:
            outcomes = [verdicts.get(task.id, False) for verdicts in per_seed]
            per_task[task.id] = {
                **stats.pass_rate(outcomes),
                "flaky": stats.is_flaky(outcomes),
                "outcomes": outcomes,
            }

        # Suite pass rate per seed, then variance across seeds. This is the
        # run-to-run noise the gate has to clear. Taking the spread across tasks
        # instead would measure how much the tasks differ from one another --
        # always large, and unrelated to whether a re-run changes the verdict.
        per_seed_rates = [
            stats.mean([1.0 if verdicts.get(t.id, False) else 0.0 for t in self.tasks])
            for verdicts in per_seed
        ]

        return {
            "arm": arm.label,
            "effect_label": arm.effect_label,
            "seeds": self.seeds,
            "per_task": per_task,
            "per_seed_pass_rates": [round(r, 4) for r in per_seed_rates],
            "suite_pass_rate": round(stats.mean(per_seed_rates), 4),
            "suite_stdev": round(stats.stdev(per_seed_rates), 4),
            "flaky_tasks": sorted(t for t, v in per_task.items() if v["flaky"]),
        }

    # -- analysis ----------------------------------------------------------

    def run(self, clean: Arm, regressions: List[Arm], output_dir: str) -> Dict[str, Any]:
        """Run the clean arm and every regression arm, then derive detection."""
        os.makedirs(output_dir, exist_ok=True)

        baseline = self.run_arm(clean)
        # Noise floor: how far the clean arm's own tasks vary. A drop smaller
        # than this is indistinguishable from the agent being itself.
        noise = baseline["suite_stdev"]

        arms = []
        detections = []
        for arm in regressions:
            observed = self.run_arm(arm)
            effect = round(baseline["suite_pass_rate"] - observed["suite_pass_rate"], 4)

            # Per-task detection: the faulted arm passed significantly less often
            # than the clean arm, by a one-tailed Fisher exact test.
            #
            # Only tasks that actually passed in the clean arm can detect
            # anything -- a task already failing has no room to regress, and
            # counting it would inflate the denominator with tasks that were
            # never capable of producing evidence.
            eligible = [
                task.id for task in self.tasks
                if baseline["per_task"][task.id]["point"] > 0.0
            ]
            detected_tasks = []
            for task_id in eligible:
                clean, faulted = baseline["per_task"][task_id], observed["per_task"][task_id]
                p_value = stats.fisher_exact_decrease(
                    clean["passed"], clean["trials"], faulted["passed"], faulted["trials"]
                )
                if faulted["point"] < clean["point"] and p_value <= self.ALPHA:
                    detected_tasks.append({"task_id": task_id, "p_value": p_value,
                                           "clean": clean["point"], "faulted": faulted["point"]})
            detection_rate = round(len(detected_tasks) / len(eligible), 4) if eligible else 0.0

            arms.append({
                **observed,
                "true_effect": effect,
                "detected_by_tasks": detected_tasks,
                "eligible_tasks": len(eligible),
                "detection_rate": detection_rate,
                # Suite-level verdict: at least one task showed a statistically
                # significant drop. The effect must also clear the clean arm's
                # own run-to-run noise, so jitter alone cannot fail a build.
                "gate_would_fail": bool(detected_tasks) and effect > noise,
            })
            detections.append({"effect": effect, "detection_rate": detection_rate})

        fired = [a for a in arms if a["gate_would_fail"]]
        report = {
            "baseline": baseline,
            "noise_floor": noise,
            "arms": arms,
            "regressions_planted": len(arms),
            "regressions_detected": len(fired),
            "suite_detection_rate": round(len(fired) / len(arms), 4) if arms else None,
            "minimum_detectable_effect": stats.minimum_detectable_effect(
                [{"effect": a["true_effect"], "detection_rate": 1.0 if a["gate_would_fail"] else 0.0}
                 for a in arms]
            ),
        }

        with open(os.path.join(output_dir, "regression_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        self._write_flaky_report(baseline, arms, output_dir)
        return report

    @staticmethod
    def _write_flaky_report(baseline, arms, output_dir: str) -> None:
        """Quarantine list: tasks whose verdict changes without any fault."""
        lines = [
            "# Flaky tasks",
            "",
            "Tasks that changed verdict across seeds. They carry no regression",
            "signal -- they move on their own -- so they are reported here rather",
            "than being allowed to shift the headline number.",
            "",
        ]
        if baseline["flaky_tasks"]:
            lines.append("## Clean arm (no faults injected)")
            lines.append("")
            for task_id in baseline["flaky_tasks"]:
                entry = baseline["per_task"][task_id]
                lines.append(f"- `{task_id}` - passed {entry['passed']}/{entry['trials']} seeds")
            lines.append("")
        else:
            lines.append("No flaky tasks in the clean arm: every task was fully")
            lines.append("deterministic across all seeds.")
            lines.append("")

        for arm in arms:
            if arm["flaky_tasks"]:
                lines.append(f"## {arm['arm']}")
                lines.append("")
                for task_id in arm["flaky_tasks"]:
                    entry = arm["per_task"][task_id]
                    lines.append(f"- `{task_id}` - passed {entry['passed']}/{entry['trials']} seeds")
                lines.append("")

        with open(os.path.join(output_dir, "flaky.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
