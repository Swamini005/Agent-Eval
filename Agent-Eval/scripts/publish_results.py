"""
Publish one CI run's result as a line of history the dashboard can read.

    python scripts/publish_results.py --out results/history.jsonl

A deployed dashboard has no access to the runner's filesystem: CI produces
reports on a throwaway machine and the Streamlit app runs somewhere else
entirely. Reading `reports/` works locally and shows nothing once deployed.

So each run appends a compact record -- scores, gate verdict, and the PR it came
from -- to a newline-delimited file kept on a separate branch. The dashboard
fetches that one file over HTTP. Append-only, so a run can never corrupt earlier
history, and cheap to read because it is a single request.

Only the summary is published, never traces or responses: the history file is
public for a public repo, and prompts or agent output could carry anything.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone

from app.config import REPORTS_DIR


def read_json(name, default=None):
    path = os.path.join(REPORTS_DIR, name)
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def git(*args, default=""):
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def ci_context():
    """
    Where this run came from.

    Values are taken from the GitHub Actions environment when present and fall
    back to local git, so the same script works on a runner and on a laptop.
    """
    event = os.getenv("GITHUB_EVENT_NAME", "local")
    ref = os.getenv("GITHUB_HEAD_REF") or os.getenv("GITHUB_REF_NAME") or git("rev-parse", "--abbrev-ref", "HEAD", default="unknown")

    pr_number = None
    event_path = os.getenv("GITHUB_EVENT_PATH")
    if event_path and os.path.exists(event_path):
        with open(event_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        pr_number = (payload.get("pull_request") or {}).get("number")

    return {
        "event": event,
        "branch": ref,
        "pr_number": pr_number,
        "commit": (os.getenv("GITHUB_SHA") or git("rev-parse", "HEAD", default=""))[:8],
        "run_id": os.getenv("GITHUB_RUN_ID"),
        "run_url": (
            f"{os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{os.getenv('GITHUB_REPOSITORY', '')}/actions/runs/{os.getenv('GITHUB_RUN_ID', '')}"
            if os.getenv("GITHUB_RUN_ID") else None
        ),
        "actor": os.getenv("GITHUB_ACTOR"),
    }


def build_record(gate_passed):
    summary = read_json("evaluation_summary.json")
    if summary is None:
        # Publishing runs with if:always(), so it also runs when the pipeline
        # never got far enough to write a report. Failing here adds a second red
        # step that names the wrong cause and buries the real one. Nothing to
        # publish is not an error in this script.
        print(f"No evaluation_summary.json in {REPORTS_DIR}/; nothing to publish.")
        raise SystemExit(0)

    agent = (read_json("agent_report.json", {}) or {}).get("agent_performance", {})
    execution = (read_json("execution.json", {}) or {}).get("summary", {})
    regression = read_json("regression_report.json")
    ablation = read_json("ablation_report.json")
    triage = read_json("triage.json")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ci": ci_context(),
        "run": summary.get("run", {}),
        "gate_passed": gate_passed,
        "global_average_score": summary.get("global_average_score"),
        "total_tasks_evaluated": summary.get("total_tasks_evaluated"),
        "summary_metrics": summary.get("summary_metrics", {}),
        "regression_catch_rate": summary.get("regression_catch_rate", {}),
        "adversarial_refusal_rate": summary.get("adversarial_refusal_rate"),
        "performance": {
            "average_latency_seconds": agent.get("average_latency_seconds"),
            "average_cost_usd": agent.get("average_cost_usd"),
            "cost_per_successful_task_usd": agent.get("cost_per_successful_task_usd"),
            "average_total_tokens": agent.get("average_total_tokens"),
            "pricing_model": agent.get("pricing_model"),
            "token_source": agent.get("token_source"),
            "tasks_passed": agent.get("tasks_passed"),
        },
        # A run stopped by its token budget is partial; the dashboard must not
        # plot it as a complete measurement.
        "complete": execution.get("complete", True),
        "budget_stop_reason": execution.get("budget_stop_reason"),
    }

    # Present only when the heavier jobs ran, so the dashboard can show them
    # without every row having to carry empty fields.
    if regression:
        record["instrument"] = {
            "minimum_detectable_effect": regression.get("minimum_detectable_effect"),
            "regressions_planted": regression.get("regressions_planted"),
            "regressions_that_degraded": regression.get("regressions_that_degraded"),
            "degrading_detection_rate": regression.get("degrading_detection_rate"),
        }
    if triage:
        record["triage"] = {
            "suite_size": triage.get("suite_size"),
            "selected": len(triage.get("task_ids", [])),
            "mandatory": len(triage.get("mandatory", [])),
            "suggested": len(triage.get("suggested", [])),
            "advisor_status": triage.get("advisor_status"),
            "rule_reason": triage.get("rule_reason"),
            "rejected": triage.get("rejected_suggestions", []),
        }
    if ablation:
        record["ablation"] = {
            "total_improvement": ablation.get("total_improvement"),
            "baseline": ablation.get("baseline"),
            "variants": [
                {"variant": v["variant"], "pass_rate": v["pass_rate"],
                 "delta": v.get("delta_vs_baseline"), "verdict": v.get("verdict")}
                for v in ablation.get("variants", [])
            ],
        }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/history.jsonl",
                        help="History file to append to")
    parser.add_argument("--gate-passed", choices=["true", "false"], default="true",
                        help="Whether the CI gate passed for this run")
    args = parser.parse_args()

    record = build_record(args.gate_passed == "true")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    ci = record["ci"]
    label = f"PR #{ci['pr_number']}" if ci["pr_number"] else ci["branch"]
    print(f"Appended {label} @ {ci['commit']} "
          f"(score={record['global_average_score']}, gate={'pass' if record['gate_passed'] else 'FAIL'}) "
          f"to {args.out}")


if __name__ == "__main__":
    main()
