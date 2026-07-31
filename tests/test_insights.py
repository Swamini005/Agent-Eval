"""The insight engine: a statement, not a table of numbers.

The rule it must never break is that a drop is only called a regression when a
Fisher exact test says it is unlikely to be noise. A dashboard that manufactures
alarm from one unlucky run is worse than no dashboard.
"""

from app.evaluation.insights import compare_runs, find_reference


def run(passed, total, *, branch="main", pr=None, gate=True, complete=True,
        metrics=None, perf=None, instrument=None):
    return {
        "ci": {"branch": branch, "pr_number": pr, "commit": "abc12345"},
        "gate_passed": gate,
        "complete": complete,
        "total_tasks_evaluated": total,
        "global_average_score": round(passed / total, 3),
        "summary_metrics": metrics or {},
        "regression_catch_rate": {"overall": 1.0},
        "performance": {"tasks_passed": passed, "token_source": "provider", **(perf or {})},
        **({"instrument": instrument} if instrument else {}),
    }


def titles(cards):
    return " | ".join(c["title"] for c in cards)


def test_a_large_significant_drop_is_called_a_regression():
    cards = compare_runs(run(10, 30), run(28, 30))
    top = cards[0]
    assert top["severity"] == "critical"
    assert "Pass rate dropped" in top["title"]
    assert "unlikely to be sampling noise" in top["detail"]


def test_a_small_drop_on_few_tasks_is_reported_as_noise():
    """The guard against manufacturing alarm: same direction, not significant."""
    cards = compare_runs(run(4, 6), run(5, 6))
    pass_cards = [c for c in cards if c["metric"] == "pass_rate"]
    assert pass_cards and pass_cards[0]["severity"] == "info"
    assert "not distinguishable from noise" in pass_cards[0]["detail"].lower()


def test_an_improvement_is_reported_with_its_significance():
    cards = compare_runs(run(29, 30), run(11, 30))
    top = [c for c in cards if c["metric"] == "pass_rate"][0]
    assert top["severity"] == "good"
    assert "improved" in top["title"]


def test_a_safety_drop_is_always_critical():
    cards = compare_runs(
        run(28, 30, metrics={"safety_and_policy": 0.8}),
        run(28, 30, metrics={"safety_and_policy": 1.0}),
    )
    safety = [c for c in cards if c["metric"] == "safety_and_policy"][0]
    assert safety["severity"] == "critical"
    assert "safety must never regress" in safety["detail"]


def test_incomplete_and_failing_runs_are_surfaced():
    cards = compare_runs(run(3, 30, gate=False, complete=False), run(28, 30))
    assert "Run did not complete" in titles(cards)
    assert "CI gate failed" in titles(cards)


def test_estimated_cost_is_flagged_so_it_is_not_read_as_measured():
    current = run(28, 30, perf={"token_source": "estimated"})
    assert "Cost figures are estimated" in titles(compare_runs(current, run(28, 30)))


def test_unmeasured_regression_detection_is_surfaced():
    current = run(28, 30)
    current["regression_catch_rate"] = {"overall": None}
    assert "Regression detection not measured" in titles(compare_runs(current, run(28, 30)))


def test_cost_and_latency_are_reported_as_ratios():
    cards = compare_runs(
        run(28, 30, perf={"cost_per_successful_task_usd": 0.02, "average_latency_seconds": 12.0}),
        run(28, 30, perf={"cost_per_successful_task_usd": 0.01, "average_latency_seconds": 10.0}),
    )
    cost = [c for c in cards if c["metric"] == "cost_per_successful_task_usd"]
    assert cost and "up 100%" in cost[0]["title"]
    # A 20% latency move is at the reporting floor and must not be dropped.
    assert any(c["metric"] == "average_latency_seconds" for c in cards)


def test_no_reference_reports_absolute_checks_only():
    cards = compare_runs(run(28, 30), None)
    assert "No earlier run to compare against" in titles(cards)
    assert not any(c["metric"] == "pass_rate" for c in cards)


def test_the_most_severe_finding_comes_first():
    cards = compare_runs(
        run(10, 30, gate=False, metrics={"safety_and_policy": 0.5}),
        run(28, 30, metrics={"safety_and_policy": 1.0}),
    )
    assert cards[0]["severity"] == "critical"


def test_a_pull_request_is_compared_against_main_not_its_own_previous_run():
    history = [
        run(28, 30, branch="main"),
        run(20, 30, branch="feature", pr=7),
        run(21, 30, branch="feature", pr=7),
    ]
    reference = find_reference(history)
    assert reference["ci"]["branch"] == "main"


def test_a_branch_run_is_compared_against_the_previous_run_on_that_branch():
    history = [run(28, 30, branch="main"), run(25, 30, branch="main")]
    assert find_reference(history) is history[0]
