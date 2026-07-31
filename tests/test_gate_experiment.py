"""The gate that fails when the *test suite* regresses, not the agent.

No eval framework ships this check. A suite that has silently lost the ability
to detect a regression still reports green, and every build after that point is
uninformative.
"""

import json
import os

from app.evaluation.gate_check import check_experiment

THRESHOLDS = {
    "min_pass_rate_lower_bound": 0.65,
    "must_detect": ["random_tool_failure"],
    "max_minimum_detectable_effect": 0.30,
    "max_flaky_tasks": 2,
}


def write_report(tmp_path, **overrides):
    report = {
        "baseline": {
            "suite_pass_rate": 0.867,
            "suite_stdev": 0.0,
            "per_seed_pass_rates": [0.867] * 5,
            "seeds": [0, 1, 2, 3, 4],
            "flaky_tasks": [],
        },
        "arms": [{"arm": "random_tool_failure", "true_effect": 0.233, "gate_would_fail": True}],
        "minimum_detectable_effect": 0.233,
    }
    report.update(overrides)
    path = os.path.join(tmp_path, "regression_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f)
    return path


def test_healthy_experiment_passes(tmp_path):
    assert check_experiment(THRESHOLDS, write_report(str(tmp_path))) == []


def test_absent_report_is_skipped_not_failed(tmp_path):
    """A single-run pipeline cannot support these checks; it must not be blocked."""
    assert check_experiment(THRESHOLDS, os.path.join(str(tmp_path), "missing.json")) == []


def test_suite_losing_detection_fails_the_build(tmp_path):
    """The headline check: the agent is unchanged, the suite has gone blind."""
    path = write_report(str(tmp_path), arms=[
        {"arm": "random_tool_failure", "true_effect": 0.0, "gate_would_fail": False}
    ])
    violations = check_experiment(THRESHOLDS, path)

    assert len(violations) >= 1
    assert "NO LONGER DETECTS" in violations[0]
    assert "test suite has regressed, not the agent" in violations[0]


def test_regression_never_run_fails(tmp_path):
    path = write_report(str(tmp_path), arms=[])
    violations = check_experiment(THRESHOLDS, path)
    assert any("never run" in v for v in violations)


def test_blunter_instrument_fails(tmp_path):
    path = write_report(str(tmp_path), minimum_detectable_effect=0.45)
    assert any("blunter" in v for v in check_experiment(THRESHOLDS, path))


def test_unresolvable_instrument_fails(tmp_path):
    path = write_report(str(tmp_path), minimum_detectable_effect=None)
    assert any("no measurable resolution" in v for v in check_experiment(THRESHOLDS, path))


def test_gate_uses_the_lower_bound_not_the_point_estimate(tmp_path):
    """A noisy run whose mean clears the bar must still fail on its lower bound."""
    noisy = {
        "suite_pass_rate": 0.70, "suite_stdev": 0.10,
        "per_seed_pass_rates": [0.6, 0.8, 0.7], "seeds": [0, 1, 2], "flaky_tasks": [],
    }
    violations = check_experiment(THRESHOLDS, write_report(str(tmp_path), baseline=noisy))

    # Point estimate 0.70 clears 0.65; lower bound 0.70 - 2*0.10 = 0.50 does not.
    assert any("lower bound" in v for v in violations)


def test_too_many_flaky_tasks_fails(tmp_path):
    path = write_report(str(tmp_path), baseline={
        "suite_pass_rate": 0.867, "suite_stdev": 0.0, "per_seed_pass_rates": [0.867],
        "seeds": [0], "flaky_tasks": ["a", "b", "c"],
    })
    assert any("flaky" in v for v in check_experiment(THRESHOLDS, path))
