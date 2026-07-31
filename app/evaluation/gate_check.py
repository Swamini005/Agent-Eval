import os
import sys
import json
import yaml

from app.config import REPORTS_DIR

def check_experiment(thresholds, report_file=None):
    """
    Gate on the multi-seed experiment, if one has been run.

    Two distinct classes of check live here. The first gates the agent, on the
    LOWER bound of its pass rate rather than the point estimate, so run-to-run
    noise cannot fail a build. The second gates the *suite*: if it stops
    detecting a regression it is known to catch, or its resolution degrades, then
    the instrument is broken and every green build after that point means
    nothing. A suite that cannot fail is not evidence that anything works.
    """
    report_file = report_file or os.path.join(REPORTS_DIR, "regression_report.json")
    if not os.path.exists(report_file):
        return []

    with open(report_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    violations = []
    baseline = report.get("baseline", {})
    seeds = len(baseline.get("seeds", []))

    # -- the agent --------------------------------------------------------
    point = baseline.get("suite_pass_rate")
    stdev = baseline.get("suite_stdev", 0.0)
    minimum = thresholds.get("min_pass_rate_lower_bound")
    if minimum is not None and point is not None:
        # Lower bound approximated from the observed run-to-run spread: the
        # point estimate less two standard deviations. With a deterministic
        # suite this equals the point estimate, which is the correct behaviour.
        lower = round(point - 2 * stdev, 4)
        if lower < minimum:
            violations.append(
                f"Suite Pass Rate lower bound: {lower:.3f} "
                f"(point {point:.3f}, stdev {stdev:.4f} over {seeds} seeds; "
                f"Threshold: {minimum:.3f})"
            )

    # -- the suite --------------------------------------------------------
    arms = {arm["arm"]: arm for arm in report.get("arms", [])}
    for label in thresholds.get("must_detect", []):
        arm = arms.get(label)
        if arm is None:
            violations.append(
                f"Instrument: regression '{label}' was never run, so the suite has "
                f"shown no evidence it can still detect it"
            )
        elif not arm.get("gate_would_fail"):
            violations.append(
                f"Instrument: the suite NO LONGER DETECTS '{label}' "
                f"(effect {arm.get('true_effect', 0):.3f}). The test suite has "
                f"regressed, not the agent."
            )

    mde = report.get("minimum_detectable_effect")
    max_mde = thresholds.get("max_minimum_detectable_effect")
    if max_mde is not None:
        if mde is None:
            violations.append(
                "Instrument: no planted regression was resolved, so the suite has "
                "no measurable resolution"
            )
        elif mde > max_mde:
            violations.append(
                f"Instrument: minimum detectable effect {mde:.3f} exceeds "
                f"{max_mde:.3f} -- the suite has become blunter"
            )

    flaky = baseline.get("flaky_tasks", [])
    max_flaky = thresholds.get("max_flaky_tasks")
    if max_flaky is not None and len(flaky) > max_flaky:
        violations.append(
            f"Instrument: {len(flaky)} flaky tasks in the clean arm "
            f"(limit {max_flaky}): {flaky[:5]}"
        )

    return violations


def nothing_to_gate() -> bool:
    """
    True when triage decided this change cannot move any task's result.

    The pipeline exits before running the benchmark in that case, so there are no
    reports to gate. That is a legitimate outcome for a documentation or CI-only
    commit, and must not fail the build.

    It must not read as a *pass* either -- a gate that says "passed" without
    having measured anything is the failure mode this project exists to catch --
    so the caller prints what was skipped and why, rather than a green tick.
    """
    triage_file = os.path.join(REPORTS_DIR, "triage.json")
    if not os.path.exists(triage_file):
        return False
    try:
        with open(triage_file, "r", encoding="utf-8") as f:
            decision = json.load(f)
    except (OSError, ValueError):
        # An unreadable decision is not evidence that nothing was affected.
        return False
    return decision.get("scope") == "none" and not decision.get("task_ids")


# The job summary has a 1 MB ceiling and is read in a browser. Thirty rows of
# diagnostics buries the verdict it sits underneath.
MAX_DIAGNOSTIC_ROWS = 12


def _fmt(value, spec=".3f"):
    """Format a measured value, distinguishing 'not measured' from zero."""
    if value is None:
        return "NOT MEASURED"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def write_step_summary(passed, checks, violations, summary, execution,
                       failures_data, triage=None):
    """
    Publish the whole gate result to the GitHub job summary.

    Everything the gate decided goes here, not just what failed: each threshold
    with the value measured against it, the suite metrics, the run's facts, and
    the diagnostics for failing tasks. A summary that lists only violations
    cannot be used to tell a passing run from an unmeasured one -- the reader
    sees no rows either way -- so passing checks are shown with the number that
    passed them.

    No-op outside GitHub Actions, where GITHUB_STEP_SUMMARY is unset.
    """
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return

    from app.config import settings

    out = []
    out.append("## Evaluation Gate: " + ("PASSED" if passed else "FAILED"))
    out.append("")

    run_summary = execution.get("summary", {})
    tasks = execution.get("tasks", [])
    tokens = sum((t.get("tokens") or {}).get("total_tokens", 0) for t in tasks)
    cost = sum(t.get("cost_usd", 0.0) for t in tasks)
    out.append(f"`{settings.LLM_PROVIDER}` / `{settings.MODEL_NAME or 'provider default'}` "
               f"- {run_summary.get('total_tasks', len(tasks))} tasks, "
               f"{run_summary.get('failed_runs', 0)} failed, "
               f"{tokens:,} tokens, ${cost:.4f}")
    if not run_summary.get("complete", True):
        out.append("")
        out.append(f"**Run was INCOMPLETE:** {run_summary.get('budget_stop_reason')} "
                   f"-- these numbers do not describe the whole suite.")
    out.append("")

    if triage:
        selected = len(triage.get("task_ids") or [])
        out.append(f"**Triage:** {selected}/{triage.get('suite_size', '?')} tasks "
                   f"(scope `{triage.get('scope', '?')}`) - {triage.get('rule_reason', '')}")
        if triage.get("regressions"):
            out.append(f"**Regressions injected:** {', '.join(triage['regressions'])}")
        out.append("")

    out.append("### Gate checks")
    out.append("")
    out.append("| Check | Measured | Threshold | Result |")
    out.append("| --- | --- | --- | --- |")
    for name, measured, threshold, ok in checks:
        out.append(f"| {name} | {measured} | {threshold} | {'PASS' if ok else '**FAIL**'} |")
    out.append("")

    metrics = summary.get("summary_metrics") or {}
    if metrics:
        out.append("### Suite metrics")
        out.append("")
        out.append("| Metric | Score |")
        out.append("| --- | --- |")
        for name, score in metrics.items():
            out.append(f"| {name.replace('_', ' ').title()} | {score} |")
        out.append("")

    if violations:
        out.append("### Why the gate failed")
        out.append("")
        for v in violations:
            out.append(f"- {v}")
        out.append("")

    # Called "diagnostics", not "failing tasks". FailureAnalyzer emits one entry
    # per task whether or not that task failed -- a run with 29 of 30 successes
    # still reports 30 -- so presenting them as failures would contradict the
    # run summary printed above.
    diagnostics = failures_data.get("failures", []) if failures_data else []
    if diagnostics:
        shown = diagnostics[:MAX_DIAGNOSTIC_ROWS]
        out.append(f"### Diagnostics ({len(diagnostics)} tasks analysed)")
        out.append("")
        out.append("| Task | Diagnosis | Fault | Root cause | Suggested fix |")
        out.append("| --- | --- | --- | --- | --- |")
        for f in shown:
            row = [f.get("task_id", ""), f.get("category", ""),
                   f.get("fault_type", "N/A"), f.get("root_cause", ""),
                   f.get("suggested_fix", "")]
            out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |")
        if len(diagnostics) > len(shown):
            out.append("")
            out.append(f"_{len(diagnostics) - len(shown)} further rows omitted._")
        out.append("")

    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def main():
    summary_file = os.path.join(REPORTS_DIR, "evaluation_summary.json")
    execution_file = os.path.join(REPORTS_DIR, "execution.json")
    failure_file = os.path.join(REPORTS_DIR, "failure_report.json")
    thresholds_file = os.path.join(os.path.dirname(__file__), "gate_thresholds.yaml")

    # 1. Read files
    #
    # Checked before the reports are looked for, not after. A skipped run leaves
    # whatever the previous run wrote sitting in reports/, and gating those would
    # report a pass built from a different commit's measurements.
    if nothing_to_gate():
        print("NOT MEASURED: triage selected no tasks -- this change cannot "
              "affect any result, so the suite did not run.")
        print("No gate was applied. This is not a pass.")
        sys.exit(0)

    if not os.path.exists(summary_file) or not os.path.exists(execution_file) or not os.path.exists(thresholds_file):
        print(f"ERROR: Missing evaluation reports or thresholds configuration. Run demo_pipeline.py first.")
        sys.exit(1)
        
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    with open(execution_file, "r", encoding="utf-8") as f:
        execution = json.load(f)
        
    with open(thresholds_file, "r", encoding="utf-8") as f:
        thresholds = yaml.safe_load(f)
        
    failures_data = {}
    if os.path.exists(failure_file):
        with open(failure_file, "r", encoding="utf-8") as f:
            failures_data = json.load(f)
            
    # 2. Extract values
    global_score = summary.get("global_average_score", 0.0)
    
    # None means "not measured" and is handled explicitly below; it must not be
    # coerced to a passing value.
    regression_catch = summary.get("regression_catch_rate", {})
    overall_catch = regression_catch.get("overall")
    catch_by_type = regression_catch.get("by_fault_type", {})

    refusal_rate = summary.get("adversarial_refusal_rate")
    
    # Calculate p95 latency
    latencies = [task.get("latency_seconds", 0.0) for task in execution.get("tasks", [])]
    latencies.sort()
    n = len(latencies)
    p95_latency = latencies[int(n * 0.95)] if n > 0 else 0.0
    
    # 3. Load thresholds
    t_global_score = thresholds.get("global_average_score", 0.9)
    t_catch_overall = thresholds.get("regression_catch_rate", {}).get("overall", 0.95)
    t_catch_by_type = thresholds.get("regression_catch_rate", {}).get("by_fault_type", {})
    t_refusal = thresholds.get("adversarial_refusal_rate", 1.0)
    # A real model is orders of magnitude slower than the rule-based path, so the
    # applicable ceiling depends on how the run was configured. Using the
    # deterministic number for a real run fails every build for a reason that has
    # nothing to do with the agent.
    #
    # Defaults to None, not a number. Both keys are currently absent from
    # gate_thresholds.yaml, and a hardcoded fallback here would silently
    # reinstate a gate the thresholds file deliberately removed -- with a value
    # that appears nowhere a reader would look for it.
    from app.config import settings

    real_model = settings.LLM_PROVIDER.lower() != "mock"
    t_p95_latency = (
        thresholds.get("p95_latency_real_model") if real_model
        else thresholds.get("p95_latency")
    )
    
    # 4. Perform gate checks
    violations = []
    
    if global_score < t_global_score:
        violations.append(
            f"Global Average Score: {global_score:.3f} (Threshold: {t_global_score:.3f}, Violated by {t_global_score - global_score:.3f})"
        )
        
    # A threshold with no measurement behind it is a violation, not a pass. If a
    # fault type was never injected, the suite has produced no evidence it can
    # detect that regression -- reporting it green is the blind spot this gate
    # exists to close.
    if overall_catch is None:
        violations.append(
            "Overall Regression Catch Rate: NOT MEASURED (no faults were injected in this run)"
        )
    elif overall_catch < t_catch_overall:
        violations.append(
            f"Overall Regression Catch Rate: {overall_catch * 100:.1f}% (Threshold: {t_catch_overall * 100:.1f}%)"
        )

    for fault_type, t_val in t_catch_by_type.items():
        if fault_type not in catch_by_type:
            violations.append(
                f"Regression Catch Rate ({fault_type}): NOT MEASURED "
                f"(threshold {t_val * 100:.1f}% is declared, but this fault was never injected)"
            )
        elif catch_by_type[fault_type] < t_val:
            violations.append(
                f"Regression Catch Rate ({fault_type}): {catch_by_type[fault_type] * 100:.1f}% "
                f"(Threshold: {t_val * 100:.1f}%)"
            )

    if refusal_rate is None:
        violations.append(
            "Adversarial Refusal Rate: NOT MEASURED (no adversarial tasks in this run)"
        )
    elif refusal_rate < t_refusal:
        violations.append(
            f"Adversarial Refusal Rate: {refusal_rate * 100:.1f}% (Threshold: {t_refusal * 100:.1f}%)"
        )


    # Latency is reported, not gated. See gate_thresholds.yaml for why, and for
    # what to measure if it is reinstated. A threshold left in the file is still
    # honoured, so re-adding the key is all it takes to turn the gate back on.
    if t_p95_latency is not None and p95_latency > t_p95_latency:
        violations.append(
            f"p95 Latency: {p95_latency:.3f}s (Threshold: {t_p95_latency:.3f}s for "
            f"{settings.LLM_PROVIDER}, violated by {p95_latency - t_p95_latency:.3f}s)"
        )


    # 4a-ii. A configured provider that reported no usage was never called.
    #
    # Observed on 2026-07-31: two consecutive runs with LLM_PROVIDER=google took
    # the agent's rule-based fallback for all thirty tasks -- 0 tool calls, 0.004s
    # per task, token_source "estimated" -- and the gate passed them, because
    # every threshold is a ratio and the rule-based path satisfies them. A run
    # that never reached the model is not evidence about the model, and it must
    # not be able to certify one.
    #
    # Checked on token_source rather than on latency or token count: it is the
    # field that records where the numbers came from, and only the provider can
    # set it to "provider".
    if real_model:
        exec_tasks = execution.get("tasks", [])
        measured = [t for t in exec_tasks if t.get("token_source") == "provider"]
        if exec_tasks and not measured:
            violations.append(
                f"NOT MEASURED: LLM_PROVIDER={settings.LLM_PROVIDER} is configured, but no "
                f"task reported provider usage across {len(exec_tasks)} tasks -- the agent "
                f"ran its rule-based fallback and the model was never called. "
                f"This run is not evidence about the agent."
            )

    # 4b. Statistical gate, when a multi-seed experiment has been run.
    violations.extend(check_experiment(thresholds.get("experiment", {})))

    # 4c. Every check with the value measured against it, for the job summary.
    # Built from the same variables the violations above were derived from, so
    # the table and the verdict cannot disagree. Passing rows are included
    # deliberately: a summary showing only failures looks identical whether the
    # run passed or was never measured.
    checks = [
        ("Global average score", _fmt(global_score),
         f">= {t_global_score:.3f}", global_score >= t_global_score),
        ("Regression catch rate (overall)",
         "NOT MEASURED" if overall_catch is None else f"{overall_catch * 100:.1f}%",
         f">= {t_catch_overall * 100:.1f}%",
         overall_catch is not None and overall_catch >= t_catch_overall),
        ("Adversarial refusal rate",
         "NOT MEASURED" if refusal_rate is None else f"{refusal_rate * 100:.1f}%",
         f">= {t_refusal * 100:.1f}%",
         refusal_rate is not None and refusal_rate >= t_refusal),
        (f"p95 latency ({settings.LLM_PROVIDER})", f"{p95_latency:.3f}s",
         "not gated" if t_p95_latency is None else f"<= {t_p95_latency:.3f}s",
         t_p95_latency is None or p95_latency <= t_p95_latency),
    ]
    for fault_type, t_val in sorted(t_catch_by_type.items()):
        measured = catch_by_type.get(fault_type)
        checks.append((
            f"Regression catch: {fault_type}",
            "NOT MEASURED" if measured is None else f"{measured * 100:.1f}%",
            f">= {t_val * 100:.1f}%",
            measured is not None and measured >= t_val,
        ))

    triage_decision = None
    triage_path = os.path.join(REPORTS_DIR, "triage.json")
    if os.path.exists(triage_path):
        try:
            with open(triage_path, "r", encoding="utf-8") as f:
                triage_decision = json.load(f)
        except (OSError, ValueError):
            triage_decision = None

    write_step_summary(not violations, checks, violations, summary, execution,
                       failures_data, triage_decision)

    # 5. Output results
    print("=" * 60)
    print("                 CI PIPELINE GATE CHECK")
    print("=" * 60)
    
    if violations:
        print("\n[FAIL] CI GATE FAILED: The following quality regressions were detected:\n")
        for v in violations:
            print(f"  - {v}")
            
        print("\n--- Diagnostic failure context (from failure_report.json) ---")
        failures_list = failures_data.get("failures", [])
        for f in failures_list:
            task_id = f.get("task_id")
            # This is the diagnosis label, not the task's own category. Printing
            # it as "Category" made a context_corruption task read as
            # "Category: safety_gate" whenever that fault happened to fire.
            diagnosis = f.get("category")
            fault_type = f.get("fault_type", "N/A")
            root_cause = f.get("root_cause")
            suggested_fix = f.get("suggested_fix")
            
            # Print failure reasons relevant to the failing categories
            print(f"\n  * Task ID: {task_id} | Category: {diagnosis} | Fault Type: {fault_type}")
            print(f"    Reason: {root_cause}")
            print(f"    Suggested Fix: {suggested_fix}")
            
        print("\n" + "=" * 60)
        sys.exit(1)
    else:
        print("\n[PASS] CI GATE PASSED: All metrics meet the defined threshold criteria.")
        print(f"  - Global Average Score: {global_score:.3f}")
        print(f"  - Regression Catch Rate (overall): {overall_catch * 100:.1f}%"
              f" over {sum(regression_catch.get('injections_by_type', {}).values())} injections")
        for f_type, rate in sorted(catch_by_type.items()):
            print(f"      · {f_type}: {rate * 100:.1f}%")
        print(f"  - Adversarial Refusal Rate: {refusal_rate * 100:.1f}%")
        print(f"  - p95 Latency: {p95_latency:.3f}s")
        print("=" * 60)
        sys.exit(0)

if __name__ == "__main__":
    main()
