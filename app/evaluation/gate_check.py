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


def main():
    summary_file = os.path.join(REPORTS_DIR, "evaluation_summary.json")
    execution_file = os.path.join(REPORTS_DIR, "execution.json")
    failure_file = os.path.join(REPORTS_DIR, "failure_report.json")
    thresholds_file = os.path.join(os.path.dirname(__file__), "gate_thresholds.yaml")
    
    # 1. Read files
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
    from app.config import settings

    real_model = settings.LLM_PROVIDER.lower() != "mock"
    t_p95_latency = (
        thresholds.get("p95_latency_real_model", 60.0) if real_model
        else thresholds.get("p95_latency", 5.0)
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


    if p95_latency > t_p95_latency:
        violations.append(
            f"p95 Latency: {p95_latency:.3f}s (Threshold: {t_p95_latency:.3f}s for "
            f"{settings.LLM_PROVIDER}, violated by {p95_latency - t_p95_latency:.3f}s)"
        )
        
    # 4b. Statistical gate, when a multi-seed experiment has been run.
    violations.extend(check_experiment(thresholds.get("experiment", {})))

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
            print(f"\n  * Task ID: {task_id} | Category: {category} | Fault Type: {fault_type}")
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
