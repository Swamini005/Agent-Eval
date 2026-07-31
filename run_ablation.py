"""
Ablation: measure each agent variant against the frozen baseline.

    python run_ablation.py [--suite=dev] [--seeds=5] [--variants=a,b,c]

Runs every variant in agents/ over the same task suite and the same seeds, then
reports each one's pass rate, its delta against the baseline, and whether that
delta clears the suite's own resolution.

A delta smaller than the minimum detectable effect measured by run_experiment.py
is not an improvement, it is noise. Reporting a gain without that comparison is
the mistake this harness exists to prevent, so the significance of every delta is
stated rather than assumed.
"""

import json
import os
import sys
import tempfile

import app.adapters  # noqa: F401  (registers the concrete adapters)
from app.adapters.factory import AgentFactory
from app.agent_variants import available_variants, load_variant
from app.benchmarks.runner import BenchmarkRunner
from app.logging_setup import configure, verbosity_from_args
from app.benchmarks.suites import load_suite
from app.config import REPORTS_DIR
from app.evaluation import statistics as stats
from app.evaluation.engine import EvaluationEngine
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput

BASELINE = "baseline"
DEFAULT_SUITE = "dev"
DEFAULT_SEEDS = 5

# Resolution of the suite, from run_experiment.py. A delta at or below this is
# indistinguishable from noise and must not be reported as an improvement.
MDE_FILE = "regression_report.json"
FALLBACK_MDE = 0.067


def load_mde() -> float:
    path = os.path.join(REPORTS_DIR, MDE_FILE)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            mde = json.load(f).get("minimum_detectable_effect")
        if mde:
            return mde
    return FALLBACK_MDE


def run_variant(variant, tasks, seeds):
    """Run one variant across every seed; return per-seed pass rates."""
    per_seed = []
    for _ in seeds:
        def build(task_id: str):
            adapter = AgentFactory.create_agent("react")
            return adapter.configure(variant)

        runner = BenchmarkRunner(build, concurrency=4, max_retries=0)
        with tempfile.TemporaryDirectory() as workdir:
            execution = runner.run_benchmark(tasks, output_dir=workdir)
            reports = EvaluationEngine().evaluate_run(
                tasks=[
                    EvaluationTaskInput(
                        task_id=t.id, benchmark=t.benchmark, category=t.category,
                        difficulty=t.difficulty,
                        domain=t.domain, prompt=t.prompt, expected_answer=t.expected_answer,
                        expected_tools=t.expected_tools, ground_truth=t.ground_truth,
                    ) for t in tasks
                ],
                executions=[
                    EvaluationExecutionInput(
                        task_id=e["task_id"], category=e.get("category", "general"),
                        response=e["response"], latency_seconds=e["latency_seconds"],
                        cost_usd=e["cost_usd"], tool_calls=e["tool_calls"], tokens=e["tokens"], token_source=e.get("token_source", "estimated"),
                        memory_state=e["memory_state"],
                        retrieval_documents=e["retrieval_documents"],
                        reasoning_nodes=e["reasoning_nodes"], errors=e["errors"],
                    ) for e in execution["tasks"]
                ],
                fault_report={"injections": []},
                output_dir=workdir,
                run_metadata={"variant": variant.name, "agent_cfg_sha": variant.sha},
            )

        outcomes = {
            r["task_id"]: bool(r["details"].get("assertions", {}).get("all_passed"))
            for r in reports["results"]
        }
        per_seed.append(outcomes)

    rates = [
        stats.mean([1.0 if o.get(t.id) else 0.0 for t in tasks]) for o in per_seed
    ]
    passed = sum(1 for t in tasks if per_seed[0].get(t.id))
    return {
        "variant": variant.name,
        "agent_cfg_sha": variant.sha,
        "description": variant.description.strip(),
        "pass_rate": round(stats.mean(rates), 4),
        "stdev": round(stats.stdev(rates), 4),
        "tasks_passed": passed,
        "tasks_total": len(tasks),
        "per_task": {t.id: per_seed[0].get(t.id, False) for t in tasks},
    }


def main():
    configure(verbosity_from_args(sys.argv))
    suite_name, seed_count = DEFAULT_SUITE, DEFAULT_SEEDS
    names = available_variants()
    for arg in sys.argv[1:]:
        if arg.startswith("--suite="):
            suite_name = arg.split("=", 1)[1]
        elif arg.startswith("--seeds="):
            seed_count = int(arg.split("=", 1)[1])
        elif arg.startswith("--variants="):
            names = [n.strip() for n in arg.split("=", 1)[1].split(",")]

    if BASELINE not in names:
        print(f"ERROR: '{BASELINE}' must be included; every delta is measured against it.")
        return 2

    suite = load_suite(suite_name)
    seeds = list(range(seed_count))
    mde = load_mde()

    # Baseline first so later variants can be compared against it.
    ordered = [BASELINE] + [n for n in names if n != BASELINE]

    print("=" * 78)
    print("                        AGENT ABLATION")
    print("=" * 78)
    print(f"suite     : {suite.name} ({len(suite)} tasks, eval_set_sha={suite.sha})")
    print(f"seeds     : {seeds}")
    print(f"variants  : {ordered}")
    print(f"resolution: {mde:.3f} pass-rate points (from the regression experiment)")
    print()

    results = []
    for name in ordered:
        variant = load_variant(name)
        print(f"running {name} ...")
        results.append(run_variant(variant, suite.tasks, seeds))

    baseline = results[0]
    print()
    print(f"{'variant':<22}{'sha':<18}{'pass rate':>10}{'passed':>9}{'delta':>9}{'verdict':>14}")
    print("-" * 82)
    for r in results:
        delta = round(r["pass_rate"] - baseline["pass_rate"], 4)
        if r is baseline:
            verdict = "baseline"
        elif delta <= 0:
            verdict = "no gain"
        elif delta <= mde:
            verdict = "within noise"
        else:
            verdict = "REAL GAIN"
        r["delta_vs_baseline"] = delta
        r["verdict"] = verdict
        print(f"{r['variant']:<22}{r['agent_cfg_sha']:<18}{r['pass_rate']:>10.3f}"
              f"{r['tasks_passed']:>6}/{r['tasks_total']:<2}{delta:>+9.3f}{verdict:>14}")
    print("-" * 82)

    best = max(results, key=lambda r: r["pass_rate"])
    total = round(best["pass_rate"] - baseline["pass_rate"], 4)
    print()
    print(f"Baseline           : {baseline['pass_rate']:.3f}  ({baseline['agent_cfg_sha']})")
    print(f"Best variant       : {best['variant']} {best['pass_rate']:.3f}  ({best['agent_cfg_sha']})")
    print(f"Total improvement  : {total:+.3f} = {total * 100:+.1f} pass-rate points")
    print(f"  {'above' if total > mde else 'within'} the suite's resolution of {mde * 100:.1f} points"
          f" -> {'a real gain' if total > mde else 'not distinguishable from noise'}")

    # Per-task attribution: which tasks each variant fixed relative to baseline.
    print()
    print("Tasks fixed relative to baseline:")
    for r in results[1:]:
        fixed = sorted(t for t, ok in r["per_task"].items() if ok and not baseline["per_task"][t])
        broke = sorted(t for t, ok in r["per_task"].items() if not ok and baseline["per_task"][t])
        print(f"  {r['variant']:<22} fixed={fixed or '[]'}")
        if broke:
            print(f"  {'':<22} BROKE={broke}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    out = os.path.join(REPORTS_DIR, "ablation_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "suite": {"name": suite.name, "eval_set_sha": suite.sha},
            "seeds": seeds,
            "resolution_mde": mde,
            "baseline": baseline["variant"],
            "total_improvement": total,
            "variants": results,
        }, f, indent=2)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
