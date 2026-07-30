import os
import yaml
import pytest

WORKFLOW_DIR = ".github/workflows"


def load_workflow(name):
    path = os.path.join(WORKFLOW_DIR, name)
    assert os.path.exists(path), f"missing workflow: {path}"
    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)
    assert content is not None
    # PyYAML parses the bare `on:` key as the boolean True.
    on_key = True if True in content else "on"
    assert on_key in content, "workflow declares no triggers"
    return content, content[on_key]


def test_ci_workflow_gates_every_change_on_main():
    """The fast lane must run on push and PR to main, and stay free of the
    container benchmark so a flaky service can never block a merge."""
    content, triggers = load_workflow("ci.yml")

    assert "push" in triggers
    assert "pull_request" in triggers
    assert "workflow_dispatch" in triggers
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["pull_request"]["branches"] == ["main"]

    job = content["jobs"]["test-and-evaluate"]
    assert job["runs-on"] == "ubuntu-latest"

    # Runs the agent with no live LLM, so the p95 latency gate measures the
    # agent rather than network variance.
    assert content["env"]["LLM_PROVIDER"] == "mock"

    steps = " ".join(step.get("run", "") for step in job["steps"])
    assert "pytest" in steps
    assert "demo_pipeline.py --mode=ci" in steps
    assert "app.evaluation.gate_check" in steps
    assert "docker" not in steps


def test_nightly_workflow_measures_the_suite_itself():
    """The multi-seed experiment runs on a schedule, never on a PR.

    It is (arms + 1) x seeds full suite runs, far too slow to gate a merge.
    """
    content, triggers = load_workflow("nightly-experiment.yml")

    assert "schedule" in triggers
    assert "workflow_dispatch" in triggers
    assert "pull_request" not in triggers

    job = content["jobs"]["experiment"]
    assert job["runs-on"] == "ubuntu-latest"
    assert content["env"]["LLM_PROVIDER"] == "mock"

    steps = " ".join(step.get("run", "") for step in job["steps"])
    assert "run_experiment.py" in steps
    # The instrument gate must run, or a suite that has gone blind still passes.
    assert "app.evaluation.gate_check" in steps


def test_workflows_declare_least_privilege_permissions():
    """Every permission granted must be one the workflow actually needs."""
    ci, _ = load_workflow("ci.yml")
    # actions:read is required to download the baseline artifact published by
    # the last successful run on main. No write scope anywhere.
    assert ci["permissions"] == {"contents": "read", "actions": "read"}

    nightly, _ = load_workflow("nightly-experiment.yml")
    assert nightly["permissions"] == {"contents": "read"}


def test_ci_compares_pull_requests_against_a_real_baseline():
    """The baseline must come from a real run on main, never be fabricated."""
    content, _ = load_workflow("ci.yml")
    steps = content["jobs"]["test-and-evaluate"]["steps"]
    by_name = {s.get("name"): s for s in steps}

    publish = by_name["Publish baseline (main only)"]
    assert "refs/heads/main" in publish["if"]

    fetch = by_name["Fetch baseline from main (PRs only)"]
    assert "pull_request" in fetch["if"]
    # A repository with no baseline yet must skip the comparison, not fail.
    assert "found=false" in fetch["run"]

    compare = by_name["Compare against main"]
    assert "steps.baseline.outputs.found == 'true'" in compare["if"]
    assert "app.evaluation.comparator" in compare["run"]

    # The old workflow generated its baseline with `echo`, so the comparison was
    # against invented numbers.
    assert "echo '{" not in " ".join(s.get("run", "") for s in steps)
