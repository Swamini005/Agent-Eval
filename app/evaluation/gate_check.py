import os
import sys
import json
import yaml

def main():
    summary_file = "evaluation_summary.json"
    execution_file = "execution.json"
    failure_file = "failure_report.json"
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
    
    regression_catch = summary.get("regression_catch_rate", {})
    overall_catch = regression_catch.get("overall", 0.0)
    catch_by_type = regression_catch.get("by_fault_type", {})
    
    refusal_rate = summary.get("adversarial_refusal_rate", 0.0)
    
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
    t_p95_latency = thresholds.get("p95_latency", 5.0)
    
    # 4. Perform gate checks
    violations = []
    
    if global_score < t_global_score:
        violations.append(
            f"Global Average Score: {global_score:.3f} (Threshold: {t_global_score:.3f}, Violated by {t_global_score - global_score:.3f})"
        )
        
    if overall_catch < t_catch_overall:
        violations.append(
            f"Overall Regression Catch Rate: {overall_catch * 100:.1f}% (Threshold: {t_catch_overall * 100:.1f}%)"
        )
        
    for fault_type, t_val in t_catch_by_type.items():
        actual_val = catch_by_type.get(fault_type, 1.0)
        if actual_val < t_val:
            violations.append(
                f"Regression Catch Rate ({fault_type}): {actual_val * 100:.1f}% (Threshold: {t_val * 100:.1f}%)"
            )
            
    if refusal_rate < t_refusal:
        violations.append(
            f"Adversarial Refusal Rate: {refusal_rate * 100:.1f}% (Threshold: {t_refusal * 100:.1f}%)"
        )
        
    if p95_latency > t_p95_latency:
        violations.append(
            f"p95 Latency: {p95_latency:.3f}s (Threshold: {t_p95_latency:.3f}s, Violated by {p95_latency - t_p95_latency:.3f}s)"
        )
        
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
            category = f.get("category")
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
        print(f"  - Regression Catch Rate (overall): {overall_catch * 100:.1f}%")
        print(f"  - Adversarial Refusal Rate: {refusal_rate * 100:.1f}%")
        print(f"  - p95 Latency: {p95_latency:.3f}s")
        print("=" * 60)
        sys.exit(0)

if __name__ == "__main__":
    main()
