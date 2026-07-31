"""Every dashboard page must render without raising.

A broken page is invisible until someone clicks it, which in practice means
during a demo. `Benchmark Explorer` shipped broken for exactly that reason: it
coloured a chart by `difficulty`, a column results.json never carried.
"""

import json
import os
import shutil
import subprocess
import sys

import pytest

from app.config import REPORTS_DIR

PAGES = [
    "Overview",
    "Benchmark Explorer",
    "Agent Leaderboard",
    "CI History (live)",
    "Regression Monitor",
    "Fault Injection Monitor",
    "Langfuse Explorer",
    "Execution Timeline",
]


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    """
    Produce a real report set for the dashboard to read.

    Any existing reports/ is moved aside and restored afterwards. This fixture
    used to delete it outright, so running the tests destroyed the artifacts of
    whatever evaluation had last been run -- including the triage decision cache,
    and including, during a demo, the very numbers on screen.

    The subprocess gets an explicit environment rather than inheriting one. It
    previously inherited .env, so a developer key turned this fixture into a live
    billed model run with a triage call in front of it.
    """
    env = {
        **os.environ,
        "LLM_PROVIDER": "mock",
        "DEEPEVAL_TELEMETRY_OPT_OUT": "YES",
        "ERROR_REPORTING": "NO",
    }

    backup = None
    if os.path.isdir(REPORTS_DIR):
        backup = str(tmp_path_factory.mktemp("saved") / "reports")
        shutil.move(REPORTS_DIR, backup)

    try:
        subprocess.run(
            [sys.executable, "demo_pipeline.py", "--suite=smoke", "--no-triage"],
            check=True, capture_output=True, env=env,
        )
        yield REPORTS_DIR
    finally:
        shutil.rmtree(REPORTS_DIR, ignore_errors=True)
        if backup:
            shutil.move(backup, REPORTS_DIR)


def app_test():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file("dashboard.py", default_timeout=120)


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_raising(page, reports):
    at = app_test()
    at.run()
    at.sidebar.selectbox[0].set_value(page).run()

    assert not at.exception, (
        f"{page} raised: {at.exception[0].value if at.exception else ''}"
    )


def test_results_carry_the_columns_the_dashboard_groups_by(reports):
    """The charts group by these; results.json dropped them at the boundary."""
    with open(f"{reports}/results.json", encoding="utf-8") as f:
        results = json.load(f)

    for column in ("category", "difficulty", "domain", "benchmark"):
        assert column in results[0], f"results.json is missing {column}"
