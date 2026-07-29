import os
import json
import sys
import argparse
from typing import Dict, Any

class BaselineComparator:
    """
    Compares evaluation metrics of a candidate run against a baseline run.
    Verifies regressions against configurable thresholds.
    Generates comparison.json and comparison.md, returning proper exit codes.
    """
    
    def __init__(self, thresholds_path: str = "thresholds.json"):
        self.thresholds = self._load_thresholds(thresholds_path)

    def _load_thresholds(self, file_path: str) -> Dict[str, float]:
        default = {
            "max_accuracy_drop": 0.05,
            "max_tool_accuracy_drop": 0.05,
            "max_latency_increase_ratio": 0.20,
            "max_cost_increase_ratio": 0.10,
            "max_hallucination_increase": 0.05,
            "max_regression_severity_increase": 0.10
        }
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return {**default, **json.load(f)}
            except Exception as e:
                print(f"Warning: Failed to load thresholds from {file_path}: {e}. Using defaults.")
        return default

    def compare(
        self,
        baseline_summary: Dict[str, Any],
        candidate_summary: Dict[str, Any],
        baseline_agent_report: Dict[str, Any],
        candidate_agent_report: Dict[str, Any],
        output_dir: str = "."
    ) -> bool:
        """
        Executes comparison checking.
        Returns True if all checks pass, False if any regressions fail thresholds.
        """
        # Helper to extract metrics safely
        base_metrics = baseline_summary.get("summary_metrics", {})
        cand_metrics = candidate_summary.get("summary_metrics", {})
        
        # Performance metrics
        base_perf = baseline_agent_report.get("agent_performance", {})
        cand_perf = candidate_agent_report.get("agent_performance", {})
        
        # Compile compared values
        # Layout: (base_val, cand_val, is_higher_better, threshold_key, threshold_val)
        comparisons = {
            "Accuracy": {
                "base": base_metrics.get("accuracy", 1.0),
                "cand": cand_metrics.get("accuracy", 1.0),
                "higher_better": True,
                "threshold_key": "max_accuracy_drop",
                "threshold": self.thresholds["max_accuracy_drop"]
            },
            "Tool Accuracy": {
                "base": base_metrics.get("tool_accuracy", 1.0),
                "cand": cand_metrics.get("tool_accuracy", 1.0),
                "higher_better": True,
                "threshold_key": "max_tool_accuracy_drop",
                "threshold": self.thresholds["max_tool_accuracy_drop"]
            },
            "Latency": {
                "base": base_perf.get("average_latency_seconds", 0.0),
                "cand": cand_perf.get("average_latency_seconds", 0.0),
                "higher_better": False,
                "threshold_key": "max_latency_increase_ratio",
                "threshold": self.thresholds["max_latency_increase_ratio"],
                "ratio_check": True
            },
            "Cost": {
                "base": base_perf.get("average_cost_usd", 0.0),
                "cand": cand_perf.get("average_cost_usd", 0.0),
                "higher_better": False,
                "threshold_key": "max_cost_increase_ratio",
                "threshold": self.thresholds["max_cost_increase_ratio"],
                "ratio_check": True
            },
            "Hallucination": {
                "base": base_metrics.get("quality", {}).get("hallucination_score", 0.0) if isinstance(base_metrics.get("quality"), dict) else 0.0,
                "cand": cand_metrics.get("quality", {}).get("hallucination_score", 0.0) if isinstance(cand_metrics.get("quality"), dict) else 0.0,
                "higher_better": False,
                "threshold_key": "max_hallucination_increase",
                "threshold": self.thresholds["max_hallucination_increase"]
            },
            "Reasoning": {
                "base": base_metrics.get("quality", {}).get("reasoning_quality", 1.0) if isinstance(base_metrics.get("quality"), dict) else 1.0,
                "cand": cand_metrics.get("quality", {}).get("reasoning_quality", 1.0) if isinstance(cand_metrics.get("quality"), dict) else 1.0,
                "higher_better": True,
                "threshold_key": "max_accuracy_drop", # reuse drop limit
                "threshold": self.thresholds["max_accuracy_drop"]
            },
            "Fault Detection": {
                "base": base_metrics.get("fault_metrics", 1.0),
                "cand": cand_metrics.get("fault_metrics", 1.0),
                "higher_better": True,
                "threshold_key": "max_accuracy_drop",
                "threshold": self.thresholds["max_accuracy_drop"]
            },
            "Regression Severity": {
                "base": base_metrics.get("fault_metrics", {}).get("regression_severity", 0.0) if isinstance(base_metrics.get("fault_metrics"), dict) else 0.0,
                "cand": cand_metrics.get("fault_metrics", {}).get("regression_severity", 0.0) if isinstance(cand_metrics.get("fault_metrics"), dict) else 0.0,
                "higher_better": False,
                "threshold_key": "max_regression_severity_increase",
                "threshold": self.thresholds["max_regression_severity_increase"]
            }
        }

        # Analyze checks
        report_results = {}
        all_passed = True
        
        for name, comp in comparisons.items():
            base_val = comp["base"]
            cand_val = comp["cand"]
            higher_better = comp["higher_better"]
            threshold = comp["threshold"]
            ratio_check = comp.get("ratio_check", False)
            
            diff = cand_val - base_val
            
            # Check regression status
            passed = True
            if higher_better:
                # Regressed if value drops more than threshold
                if diff < 0 and abs(diff) > threshold:
                    passed = False
            else:
                # Regressed if value increases more than threshold
                if ratio_check:
                    # check relative ratio increase
                    ratio = (diff / base_val) if base_val else 0.0
                    if ratio > threshold:
                        passed = False
                else:
                    if diff > threshold:
                        passed = False
                        
            if not passed:
                all_passed = False
                
            report_results[name] = {
                "baseline": round(base_val, 4),
                "candidate": round(cand_val, 4),
                "diff": round(diff, 4),
                "threshold": threshold,
                "status": "PASS" if passed else "FAIL"
            }
            
        # 1. Output comparison.json
        comp_json_file = os.path.join(output_dir, "comparison.json")
        with open(comp_json_file, "w", encoding="utf-8") as f:
            json.dump({
                "passed": all_passed,
                "metrics": report_results
            }, f, indent=2)

        # 2. Output comparison.md (pull request layout)
        comp_md_file = os.path.join(output_dir, "comparison.md")
        with open(comp_md_file, "w", encoding="utf-8") as f:
            f.write("# Branch Evaluation Comparison Report\n\n")
            f.write(f"**Overall Status:** {'✅ PASS' if all_passed else '❌ FAIL (Regression Detected)'}\n\n")
            f.write("| Metric | Baseline (main) | Candidate | Change | Threshold | Status |\n")
            f.write("|---|---|---|---|---|---|\n")
            for name, r in report_results.items():
                emoji = "✅" if r["status"] == "PASS" else "❌"
                # Sign format
                sign = "+" if r["diff"] > 0 else ""
                f.write(f"| {name} | {r['baseline']} | {r['candidate']} | {sign}{r['diff']} | {r['threshold']} | {emoji} {r['status']} |\n")
                
        print(f"Generated comparison reports: {comp_json_file} and {comp_md_file}")
        return all_passed

def main():
    parser = argparse.ArgumentParser(description="Baseline Comparison Engine CLI for CI/CD")
    parser.add_argument("--baseline-summary", default="baseline_summary.json", help="Path to baseline evaluation_summary.json")
    parser.add_argument("--candidate-summary", default="evaluation_summary.json", help="Path to current evaluation_summary.json")
    parser.add_argument("--baseline-agent", default="baseline_agent.json", help="Path to baseline agent_report.json")
    parser.add_argument("--candidate-agent", default="agent_report.json", help="Path to current agent_report.json")
    parser.add_argument("--thresholds", default="thresholds.json", help="Path to thresholds.json")
    parser.add_argument("--output-dir", default=".", help="Output directory path")
    args = parser.parse_args()

    # Load files
    try:
        with open(args.baseline_summary, "r", encoding="utf-8") as f:
            b_sum = json.load(f)
        with open(args.candidate_summary, "r", encoding="utf-8") as f:
            c_sum = json.load(f)
        with open(args.baseline_agent, "r", encoding="utf-8") as f:
            b_agent = json.load(f)
        with open(args.candidate_agent, "r", encoding="utf-8") as f:
            c_agent = json.load(f)
    except Exception as e:
        print(f"Error loading evaluation report files: {e}")
        sys.exit(1)

    comparator = BaselineComparator(thresholds_path=args.thresholds)
    passed = comparator.compare(b_sum, c_sum, b_agent, c_agent, output_dir=args.output_dir)
    
    if not passed:
        print("❌ BUILD BLOCKED: Quality regression detected exceeding threshold tolerances!")
        sys.exit(1)
    else:
        print("✅ BUILD SUCCESSFUL: Performance metrics are within tolerance parameters.")
        sys.exit(0)

if __name__ == "__main__":
    main()
