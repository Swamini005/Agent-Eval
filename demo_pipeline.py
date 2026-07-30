import json
import os
import sys
from app.benchmarks.suites import load_suite
from app.config import REPORTS_DIR
from app.adapters.factory import AgentFactory
from app.faults.engine import FaultInjectionEngine
from app.faults.middleware import FaultInjectionMiddleware
from app.benchmarks.runner import BenchmarkRunner
from app.evaluation.engine import EvaluationEngine
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput

MOCK_FAULT_CONFIG = {
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

def main():
    mode = "interactive"
    seed = DEFAULT_SEED
    suite_name = DEFAULT_SUITE
    for arg in sys.argv:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1].strip().lower()
        elif arg.startswith("--seed="):
            seed = int(arg.split("=")[1].strip())
        elif arg.startswith("--suite="):
            suite_name = arg.split("=")[1].strip().lower()

    if mode == "ci":
        print("[CI Mode] Running headless verification pipeline...")

    print("=" * 60)
    print("        START-TO-END AGENT EVALUATION PIPELINE DEMO")
    print("=" * 60)
    print("\n[NOTE] Running in offline fallback mock mode. No LLM or Langfuse API keys needed.")

    # 1. LOAD BENCHMARK
    print("\n--- 1. Loading Benchmark Tasks ---")
    suite = load_suite(suite_name)
    tasks = suite.tasks
    print(f"Suite '{suite.name}'  |  {len(suite)} tasks  |  eval_set_sha={suite.sha}")
    for category, count in suite.categories().items():
        print(f"  - {category}: {count}")

    # 2. SET UP FAULT INJECTION
    print("\n--- 2. Configuring Fault Injection ---")
    from app.faults.loader import FaultConfigLoader
    configs = FaultConfigLoader.load_from_dict(MOCK_FAULT_CONFIG["faults"])
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
            AgentFactory.create_agent("langgraph"),
            fault_engine.fork(task_id),
            domain=domains_by_task.get(task_id, "")
        )
    print("Adapter factory ready: LangGraph agent wrapped in FaultInjectionMiddleware, one per task.")

    # 4. RUN BENCHMARK TASKS (Parallel execution with captured telemetry)
    print("\n--- 4. Executing Benchmark Runner (Concurrency = 2) ---")
    runner = BenchmarkRunner(build_adapter, concurrency=2, max_retries=1)

    # We output to the local workspace
    runner.run_benchmark(tasks, output_dir=REPORTS_DIR)

    # Save the injected faults logs, aggregated across every per-task engine
    fault_engine.save_reports(workspace_path=REPORTS_DIR)
    
    print("\nSaved execution telemetry to: ./execution.json")
    print("Saved injected faults logs to: ./fault_log.json and ./fault_report.json")

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
            tokens=e["tokens"],
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
        run_metadata={"suite": suite.name, "eval_set_sha": suite.sha, "seed": seed}
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
    print("Saved diagnostics to: ./failure_report.json")

    print("\nEvaluation reports generated successfully:")
    print("  - ./results.json (detailed metric scores)")
    print("  - ./benchmark_report.json (grouped scores by benchmark provider)")
    print("  - ./agent_report.json (average speed and cost profiles)")
    print("  - ./evaluation_summary.json (global average success ratings)")
    print("  - ./failure_report.json (automatic failure diagnostics)")

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
