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
    # agent rather than network variance. A manual run may override the provider
    # for a demo, but every push and pull request must fall back to mock --
    # otherwise an unchanged commit can fail the latency gate on model variance
    # alone, and the gate stops meaning anything.
    # These live on the job, not the workflow: the `secrets` context is not
    # available in workflow-level env, so putting them there silently yields
    # empty strings and every run falls back to mock.
    job_env = job["env"]
    provider = " ".join(job_env["LLM_PROVIDER"].split())
    assert "secrets.GOOGLE_API_KEY" in provider, provider
    assert "'mock'" in provider, provider

    # A handed-over agent calls its own model rather than going through
    # app/agent/llm.py, so the key must reach the runner even on a mock run --
    # without it such an agent fails every task and reads as broken.
    assert job_env["GOOGLE_API_KEY"] == "${{ secrets.GOOGLE_API_KEY }}"

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
    # contents:write publishes the eval-results branch the deployed dashboard
    # reads; actions:read downloads the baseline artifact from the last green run
    # on main. Both are used, and nothing broader is granted.
    assert ci["permissions"] == {"contents": "write", "actions": "read"}

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


def test_ci_publishes_history_for_the_deployed_dashboard():
    """A deployed dashboard cannot read the runner's filesystem.

    Each run appends its summary to an orphan branch the app fetches over HTTP.
    """
    content, _ = load_workflow("ci.yml")
    by_name = {s.get("name"): s for s in content["jobs"]["test-and-evaluate"]["steps"]}

    publish = by_name["Publish result to the dashboard history"]
    # Forked PRs get a read-only token, so the push would fail the job.
    assert "fork != true" in publish["if"]
    assert "scripts.publish_results" in publish["run"]
    assert "eval-results" in publish["run"]


def test_a_failing_gate_is_published_before_the_build_goes_red():
    """A red run is the one most worth seeing on the dashboard.

    The gate therefore runs with continue-on-error and the job is failed
    explicitly afterwards, so publishing still happens.
    """
    content, _ = load_workflow("ci.yml")
    steps = content["jobs"]["test-and-evaluate"]["steps"]
    by_name = {s.get("name"): s for s in steps}

    assert by_name["Run CI verification gate check"]["continue-on-error"] is True

    fail_step = by_name["Fail the build if the gate failed"]
    assert "steps.gate.outcome == 'failure'" in fail_step["if"]

    order = [s.get("name") for s in steps]
    assert order.index("Publish result to the dashboard history") <            order.index("Fail the build if the gate failed")
