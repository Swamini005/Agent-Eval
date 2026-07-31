import json
import os
import sys
from app.benchmarks.suites import load_suite
from app.logging_setup import configure, verbosity_from_args
from app.budget import RunBudget
from app.config import REPORTS_DIR
from app.adapters.factory import AgentFactory
from app.faults.engine import FaultInjectionEngine
from app.faults.middleware import FaultInjectionMiddleware
from app.benchmarks.runner import BenchmarkRunner
from app.evaluation.engine import EvaluationEngine
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput

DEFAULT_FAULT_CONFIG = {
    "faults": [
        {
            "id": "FAULT-LATENCY-01",
            "type": "tool_latency",
            "component": "tool",
            "severity": "info",
            "probability": 1.0,
            "scheduling": {},
            "parameters": {"delay_seconds": 0.2}
        },
        # `after_steps` counts steps within a single task. Each task performs one
        # run() step, so a schedule above 0 would never fire. These previously
        # read 1/2/5 because the step counter was shared across every task in the
        # pool, making them trigger on whichever tasks happened to run late --
        # non-deterministically, and never attributable to a specific task.
        {
            "id": "FAULT-BYPASS-02",
            "type": "planner_bypass",
            "component": "reasoning",
            "severity": "warning",
            "probability": 1.0,
            "scheduling": {"after_steps": 0},
            "parameters": {}
        },
        {
            "id": "FAULT-CONFIRM-BYPASS-03",
            "type": "planner_bypass_confirmation",
            "component": "reasoning",
            "severity": "critical",
            "probability": 1.0,
            "scheduling": {"after_steps": 0},
            "parameters": {}
        },
        {
            "id": "FAULT-CONTEXT-CORRUPT-04",
            "type": "context_corruption",
            "component": "tool",
            "severity": "critical",
            "probability": 1.0,
            "scheduling": {"after_steps": 0},
            "parameters": {}
        }
    ]
}

# Fixed so the pipeline is reproducible: identical fault injections and therefore
# identical scores on every run. Override with --seed=N.
DEFAULT_SEED = 0

# The dev suite is the working set. Run --suite=holdout once, at the end, to
# check for overfitting; inspecting it while tuning defeats its purpose.
DEFAULT_SUITE = "dev"

# Stop before the provider does. Groq's free tier is 100,000 tokens/day and this
# agent spends ~10,000 on a task, so an unguarded 30-task run exhausts the quota
# partway through and the rest fail with 429s that read like agent failures.
# 0 disables the ceiling.
DEFAULT_MAX_TOKENS = 80_000

# Which agent to evaluate. Any adapter in app/adapters/ is discovered
# automatically, so committing one is enough to make it a valid target.
DEFAULT_TARGET = "langgraph"

# Triage picks the cases a commit needs instead of the whole suite, and is the
# default path. Every run prints what it selected and why, and writes the same to
# reports/triage.json, so a narrowed run is never silent. The rules floor is
# always included, so the gate stays defensible; the advisor can only widen it.
# Pass --no-triage to force the entire suite.
DEFAULT_TRIAGE = True

def faults_for_regressions(labels):
    """
    Fault specifications for the regressions triage chose, from regressions.yaml.

    One catalogue, read by triage to know what it may select and read here to
    know what to inject, so the two can never disagree about what a label means.
    """
    if not labels:
        return []

    import yaml

    with open("regressions.yaml", encoding="utf-8") as f:
        catalogue = {r["label"]: r for r in (yaml.safe_load(f) or {}).get("regressions", [])}

    specs = []
    for label in labels:
        entry = catalogue.get(label)
        if entry:
            specs.extend(entry.get("faults", []))
    return specs


def main():
    configure(verbosity_from_args(sys.argv))
    mode = "interactive"
    seed = DEFAULT_SEED
    suite_name = DEFAULT_SUITE
    max_tokens = DEFAULT_MAX_TOKENS
    target = DEFAULT_TARGET
    use_triage = DEFAULT_TRIAGE
    use_advisor = True
    concurrency = 2
    triaged_regressions = []
    for arg in sys.argv:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1].strip().lower()
        elif arg.startswith("--seed="):
            seed = int(arg.split("=")[1].strip())
        elif arg.startswith("--suite="):
            suite_name = arg.split("=")[1].strip().lower()
        elif arg.startswith("--max-tokens="):
            max_tokens = int(arg.split("=")[1].strip())
        elif arg.startswith("--target="):
            target = arg.split("=")[1].strip().lower()
        elif arg == "--triage":
            use_triage = True
        elif arg == "--no-triage":
            use_triage = False
        elif arg == "--no-advisor":
            use_advisor = False
        elif arg.startswith("--concurrency="):
            concurrency = int(arg.split("=")[1].strip())
        elif arg.startswith("--agent="):
            # Dotted path to a third-party agent, used with --target=external.
            # Read by ExternalAgentAdapter at construction.
            os.environ["AGENT_PATH"] = arg.split("=", 1)[1].strip()

    if mode == "ci":
        print("[CI Mode] Running headless verification pipeline...")

    print("=" * 60)
    print("        START-TO-END AGENT EVALUATION PIPELINE DEMO")
    print("=" * 60)
    # Read from configuration rather than hardcoded: this line announced mock
    # mode unconditionally, including on runs against a live model, so the
    # terminal contradicted what was actually being measured.
    from app.config import settings
    if settings.LLM_PROVIDER.lower() == "mock":
        print("\n[MODE] Deterministic mock mode: no model, no key, no network.")
    else:
        print(f"\n[MODE] Live model: provider={settings.LLM_PROVIDER} "
              f"model={settings.MODEL_NAME or '(provider default)'}")

    # 1. LOAD BENCHMARK
    print("\n--- 1. Loading Benchmark Tasks ---")
    suite = load_suite(suite_name)
    tasks = suite.tasks
    print(f"Suite '{suite.name}'  |  {len(suite)} tasks  |  eval_set_sha={suite.sha}")
    for category, count in suite.categories().items():
        print(f"  - {category}: {count}")

    # 1b. TRIAGE -- narrow the suite to the cases this commit needs
    if use_triage:
        import json as _json
        from app.benchmarks.impact import select_affected
        from app.evaluation.triage import triage, selected_tasks

        affected, rationale = select_affected(tasks)
        regression_labels = [
            r["label"] for r in
            (__import__("yaml").safe_load(open("regressions.yaml", encoding="utf-8"))
             or {}).get("regressions", [])
        ]
        history_path = os.path.join(REPORTS_DIR, "failure_history.json")
        history = _json.load(open(history_path, encoding="utf-8")) if os.path.exists(history_path) else {}

        decision = triage(
            tasks,
            mandatory_ids={t.id for t in affected},
            rule_reason=rationale.get("reason", ""),
            changed_files=rationale.get("changed_files", []),
            available_regressions=regression_labels,
            failure_history=history,
            use_advisor=use_advisor,
            # Pins the advisor's answer to this commit. Re-running the same
            # commit replays the decision instead of asking again, so the test
            # set does not change between two runs of one diff.
            cache_dir=os.path.join(REPORTS_DIR, "triage_cache"),
            suite_sha=suite.sha,
        )
        print()
        print("--- 1b. Triage ---")
        print(f"{decision.describe(len(tasks))}")
        print(f"  rules: {decision.rule_reason}")
        for task_id in sorted(decision.suggested):
            print(f"  + {task_id}: {decision.reasons.get(task_id, '')}")
        for rejected in decision.rejected:
            print(f"  ! rejected {rejected}")
        if decision.replayed:
            print("  (advisor decision replayed from cache for this commit)")

        os.makedirs(REPORTS_DIR, exist_ok=True)
        with open(os.path.join(REPORTS_DIR, "triage.json"), "w", encoding="utf-8") as f:
            _json.dump({**decision.summary(), "suite_size": len(tasks)}, f, indent=2)

        tasks = selected_tasks(tasks, decision) or tasks
        triaged_regressions = sorted(decision.regressions)

    # 2. SET UP FAULT INJECTION
    #
    # Triage selects which planted regressions this commit is worth exercising,
    # and those are what get injected. The selection was previously recorded in
    # the report and then ignored -- the pipeline always injected the same
    # hardcoded four -- so a triage decision naming a regression was decoration.
    print("\n--- 2. Configuring Fault Injection ---")
    from app.faults.loader import FaultConfigLoader

    fault_specs = faults_for_regressions(triaged_regressions)
    if fault_specs:
        print(f"Regressions selected by triage: {', '.join(triaged_regressions)}")
    else:
        # No triage, or triage named none. The default set keeps an unqualified
        # run comparable with the thresholds calibrated against it.
        fault_specs = DEFAULT_FAULT_CONFIG["faults"]

    configs = FaultConfigLoader.load_from_dict(fault_specs)
    fault_engine = FaultInjectionEngine(configs, seed=seed)
    print(f"Loaded {len(configs)} fault rules (seed={seed}).")

    # Each task declares its own domain, so a single run may span several packs.
    domains_by_task = {t.id: t.domain for t in tasks}

    # 3. RESOLVE AGENT ADAPTER PER TASK
    # Adapters and fault engines both hold per-run mutable state, so each task
    # gets its own. A shared instance interleaves traces across the thread pool
    # and misattributes fault logs to whichever task initialized last.
    print("\n--- 3. Preparing Per-Task Agent Adapters ---")
    def build_adapter(task_id: str):
        return FaultInjectionMiddleware(
            AgentFactory.create_agent(target),
            fault_engine.fork(task_id),
            domain=domains_by_task.get(task_id, "")
        )
    from app.adapters.registry import AdapterRegistry
    if target not in AdapterRegistry.available():
        raise SystemExit(
            f"Unknown target {target!r}. Discovered adapters: "
            f"{AdapterRegistry.available()}"
        )
    print(f"Target '{target}' wrapped in FaultInjectionMiddleware, one adapter per task.")

    # 4. RUN BENCHMARK TASKS (Parallel execution with captured telemetry)
    print(f"\n--- 4. Executing Benchmark Runner (Concurrency = {concurrency}) ---")
    budget = RunBudget(max_tokens=max_tokens or None)
    runner = BenchmarkRunner(build_adapter, concurrency=concurrency, max_retries=1, budget=budget)

    # We output to the local workspace
    runner.run_benchmark(tasks, output_dir=REPORTS_DIR)

    # Save the injected faults logs, aggregated across every per-task engine
    fault_engine.save_reports(workspace_path=REPORTS_DIR)
    
    print()
    print(f"Saved execution telemetry to: {REPORTS_DIR}/execution.json")
    print(f"Saved injected faults logs to: {REPORTS_DIR}/fault_log.json and {REPORTS_DIR}/fault_report.json")

    # 5. EXECUTE EVALUATION ENGINE
    print("\n--- 5. Evaluating Execution Telemetry ---")
    eval_engine = EvaluationEngine()
    
    # Load execution telemetry output
    with open(os.path.join(REPORTS_DIR, "execution.json"), "r", encoding="utf-8") as f:
        execution_data = json.load(f)
        
    # Load fault report output
    with open(os.path.join(REPORTS_DIR, "fault_report.json"), "r", encoding="utf-8") as f:
        fault_report = json.load(f)
        
    # Convert execution JSON back to Pydantic models for evaluation engine
    eval_tasks = [
        EvaluationTaskInput(
            task_id=t.id,
            benchmark=t.benchmark,
            category=t.category,
            difficulty=t.difficulty,
            domain=t.domain,
            prompt=t.prompt,
            expected_answer=t.expected_answer,
            expected_tools=t.expected_tools,
            ground_truth=t.ground_truth
        ) for t in tasks
    ]
    
    eval_executions = [
        EvaluationExecutionInput(
            task_id=e["task_id"],
            category=e.get("category", "general"),
            response=e["response"],
            latency_seconds=e["latency_seconds"],
            cost_usd=e["cost_usd"],
            tool_calls=e["tool_calls"],
            tokens=e["tokens"], token_source=e.get("token_source", "estimated"),
            memory_state=e["memory_state"],
            retrieval_documents=e["retrieval_documents"],
            reasoning_nodes=e["reasoning_nodes"],
            errors=e["errors"]
        ) for e in execution_data["tasks"]
    ]
    
    # Run evaluation
    eval_reports = eval_engine.evaluate_run(
        tasks=eval_tasks,
        executions=eval_executions,
        fault_report=fault_report,
        output_dir=REPORTS_DIR,
        run_metadata={"suite": suite.name, "eval_set_sha": suite.sha,
                      "seed": seed, "target": target}
    )
    
    # 6. Failure Analyzer
    print("\n--- 6. Running Failure Analyzer ---")
    from app.evaluation.analyzer import FailureAnalyzer
    analyzer = FailureAnalyzer()
    analyzer.analyze(
        results=eval_reports["results"],
        executions=execution_data["tasks"],
        fault_report=fault_report,
        output_dir=REPORTS_DIR
    )
    print(f"Saved diagnostics to: {REPORTS_DIR}/failure_report.json")

    print("\nEvaluation reports generated successfully:")
    print(f"  - {REPORTS_DIR}/results.json (detailed metric scores)")
    print(f"  - {REPORTS_DIR}/benchmark_report.json (grouped scores by benchmark provider)")
    print(f"  - {REPORTS_DIR}/agent_report.json (average speed and cost profiles)")
    print(f"  - {REPORTS_DIR}/evaluation_summary.json (global average success ratings)")
    print(f"  - {REPORTS_DIR}/failure_report.json (automatic failure diagnostics)")

    # Display final results summary
    print("\n" + "=" * 60)
    print("                    EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Global Average Score: {eval_reports['summary']['global_average_score']}")
    print(f"Total Tasks Evaluated: {eval_reports['summary']['total_tasks_evaluated']}")
    print("\nDetailed Metric Breakdown:")
    for metric_name, score in eval_reports['summary']['summary_metrics'].items():
        print(f"  - {metric_name.title().replace('_', ' ')}: {score}")
    print("=" * 60)

if __name__ == "__main__":
    main()
