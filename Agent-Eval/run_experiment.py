"""
Run the regression experiment: measure the evaluation suite, not the agent.

    python run_experiment.py [--suite=dev] [--seeds=3] [--target=langgraph]
                             [--replay] [--no-cache]

Results are cached per (target, suite hash, arm, seed), so re-running after a
reporting change replays instead of re-executing the agent. --replay serves only
from cache and fails on a miss, which lets a demo render with no network.

Executes a clean arm plus one arm per entry in regressions.yaml, each across
several seeds, then reports which planted regressions the suite actually caught
and the smallest effect it can resolve.
"""

import json
import os
import sys

import yaml

import app.adapters  # noqa: F401  (import registers the concrete adapters)
from app.adapters.factory import AgentFactory
from app.logging_setup import configure, verbosity_from_args
from app.benchmarks.suites import load_suite
from app.config import REPORTS_DIR
from app.evaluation.experiment import Arm, RegressionExperiment
from app.faults.models import FaultConfig

DEFAULT_SUITE = "dev"
DEFAULT_SEEDS = 3
DEFAULT_TARGET = "langgraph"


def load_catalogue(path: str = "regressions.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        catalogue = yaml.safe_load(f)
    return [
        Arm(
            label=entry["label"],
            faults=[FaultConfig(**fault) for fault in entry["faults"]],
            effect_label=entry.get("description", "").strip(),
        )
        for entry in catalogue["regressions"]
    ]


def main():
    configure(verbosity_from_args(sys.argv))
    suite_name, seed_count, target = DEFAULT_SUITE, DEFAULT_SEEDS, DEFAULT_TARGET
    replay = "--replay" in sys.argv
    no_cache = "--no-cache" in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith("--suite="):
            suite_name = arg.split("=", 1)[1].strip().lower()
        elif arg.startswith("--seeds="):
            seed_count = int(arg.split("=", 1)[1])
        elif arg.startswith("--target="):
            target = arg.split("=", 1)[1].strip().lower()

    suite = load_suite(suite_name)
    regressions = load_catalogue()
    seeds = list(range(seed_count))

    print("=" * 68)
    print("            REGRESSION EXPERIMENT - MEASURING THE SUITE")
    print("=" * 68)
    print(f"suite        : {suite.name} ({len(suite)} tasks, sha={suite.sha})")
    print(f"target       : {target}")
    print(f"seeds        : {seeds}")
    print(f"regressions  : {len(regressions)} planted")
    print(f"total runs   : {(len(regressions) + 1) * len(seeds)}")
    if replay:
        print("mode         : replay (cache only, no agent execution)")
    print()

    experiment = RegressionExperiment(
        tasks=suite.tasks,
        agent_builder=lambda: AgentFactory.create_agent(target),
        seeds=seeds,
        concurrency=4,
        target_label=target,
        suite_sha=suite.sha,
        cache_dir=None if no_cache else os.path.join(REPORTS_DIR, "cache"),
        replay=replay,
    )

    report = experiment.run(clean=Arm("clean", []), regressions=regressions, output_dir=REPORTS_DIR)
    report["suite"] = {"name": suite.name, "eval_set_sha": suite.sha, "target": target}
    with open(os.path.join(REPORTS_DIR, "regression_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    baseline = report["baseline"]
    print(f"CLEAN BASELINE  pass rate {baseline['suite_pass_rate']:.3f}"
          f"  (stdev {baseline['suite_stdev']:.4f}, {len(baseline['flaky_tasks'])} flaky)")
    print()
    print(f"{'regression':<32}{'pass rate':>10}{'effect':>9}{'detected':>10}{'gate':>8}")
    print("-" * 69)
    for arm in report["arms"]:
        print(f"{arm['arm']:<32}{arm['suite_pass_rate']:>10.3f}{arm['true_effect']:>9.3f}"
              f"{arm['detection_rate'] * 100:>9.0f}%{'FAIL' if arm['gate_would_fail'] else 'pass':>8}")
    print("-" * 69)
    print()
    print(f"Cache                     : {experiment.cache_hits} hits, {experiment.cache_misses} executed")
    degrading = report["regressions_that_degraded"]
    rate = report["degrading_detection_rate"]
    print(f"Regressions planted       : {report['regressions_planted']}")
    print(f"  of which degraded       : {degrading}"
          f"  (inert against this target: {report['inert_regressions']})")
    if rate is not None:
        print(f"  detected                : {len(
            [a for a in report['arms'] if a['gate_would_fail']])}/{degrading}"
              f"  = {rate * 100:.0f}% of regressions that actually caused harm")

    mde = report["minimum_detectable_effect"]
    if mde is None:
        print("Minimum detectable effect : none of the planted regressions were resolved")
    else:
        print(f"Minimum detectable effect : {mde:.3f} pass-rate points")
        print(f"  -> the suite reliably catches degradations of {mde * 100:.1f} points or larger")
        print(f"     at {len(seeds)} seeds. Smaller effects need more seeds.")
    print()
    print(f"Wrote {REPORTS_DIR}/regression_report.json and {REPORTS_DIR}/flaky.md")


if __name__ == "__main__":
    main()
