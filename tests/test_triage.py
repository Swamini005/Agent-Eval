"""
Triage safety: the advisor may add to the run set, never shrink it.

Every test here is about the same failure. If an LLM could remove tasks from a
run, a commit could produce a fast green build that proves nothing -- and because
the model's reasoning is not reproducible, no reviewer could tell it had happened.
So the rules floor is a floor, and each way the advisor might fail is pinned below.
"""

from unittest.mock import patch, MagicMock

import pytest

from app.benchmarks.models import UnifiedBenchmarkTask
from app.evaluation import triage as triage_mod
from app.evaluation.triage import triage, selected_tasks


def make_tasks(n=6):
    return [
        UnifiedBenchmarkTask(
            id=f"t{i}", benchmark="dev", category="multi_tool",
            domain="travel", difficulty="easy", prompt=f"prompt {i}",
        )
        for i in range(n)
    ]


FLOOR = {"t0", "t1"}


def test_advisor_cannot_remove_a_task_the_rules_required():
    """The core invariant. An empty proposal leaves the floor untouched."""
    tasks = make_tasks()
    with patch.object(triage_mod, "ask_advisor", return_value={"tasks": [], "regressions": []}):
        decision = triage(tasks, FLOOR, available_regressions=["r1"])

    assert decision.task_ids == FLOOR
    assert decision.advisor_status == "ok"


def test_advisor_proposing_a_subset_still_cannot_shrink_the_floor():
    """Even a proposal naming only one mandatory task keeps both."""
    tasks = make_tasks()
    proposal = {"tasks": [{"id": "t0", "reason": "only this one matters"}], "regressions": []}
    with patch.object(triage_mod, "ask_advisor", return_value=proposal):
        decision = triage(tasks, FLOOR)

    assert FLOOR <= decision.task_ids
    # An id already required stays attributed to the rule that required it.
    assert decision.provenance["t0"] == "rule"
    assert "t0" not in decision.suggested


def test_advisor_failure_leaves_the_floor_and_does_not_block_the_build():
    """A model outage must degrade to the rules, not to an exception."""
    tasks = make_tasks()
    with patch.object(triage_mod, "ask_advisor", side_effect=RuntimeError("503 upstream")):
        decision = triage(tasks, FLOOR)

    assert decision.task_ids == FLOOR
    assert decision.advisor_status.startswith("unavailable")
    assert "RuntimeError" in decision.advisor_status


def test_no_model_configured_is_reported_as_unavailable_not_as_a_clean_run():
    """The normal CI case. It must be visible in the report, not silent."""
    from app.agent.llm import DeterministicMode

    tasks = make_tasks()
    with patch.object(triage_mod, "ask_advisor", side_effect=DeterministicMode("mock")):
        decision = triage(tasks, FLOOR)

    assert decision.task_ids == FLOOR
    assert "DeterministicMode" in decision.advisor_status


def test_hallucinated_ids_are_rejected_and_recorded():
    """A model inventing ids is a signal about the advisor; dropping it wastes that."""
    tasks = make_tasks()
    proposal = {
        "tasks": [{"id": "t3", "reason": "real"}, {"id": "does-not-exist", "reason": "invented"}],
        "regressions": [{"label": "r1", "reason": "real"}, {"label": "no-such-fault", "reason": "invented"}],
    }
    with patch.object(triage_mod, "ask_advisor", return_value=proposal):
        decision = triage(tasks, FLOOR, available_regressions=["r1"])

    assert "t3" in decision.suggested
    assert "does-not-exist" not in decision.task_ids
    assert decision.regressions == {"r1"}
    assert any("does-not-exist" in r for r in decision.rejected)
    assert any("no-such-fault" in r for r in decision.rejected)


def test_suggestions_are_capped_so_triage_cannot_expand_back_to_the_whole_suite():
    """Given a loose brief the model suggests everything plausibly related."""
    tasks = make_tasks(n=40)
    proposal = {
        "tasks": [{"id": f"t{i}", "reason": "vague"} for i in range(2, 40)],
        "regressions": [{"label": f"r{i}", "reason": "vague"} for i in range(10)],
    }
    with patch.object(triage_mod, "ask_advisor", return_value=proposal):
        decision = triage(tasks, FLOOR, available_regressions=[f"r{i}" for i in range(10)])

    assert len(decision.suggested) <= triage_mod.MAX_SUGGESTED_TASKS
    assert len(decision.regressions) <= triage_mod.MAX_SUGGESTED_REGRESSIONS


def test_disabling_the_advisor_yields_the_rules_floor_and_says_so():
    tasks = make_tasks()
    decision = triage(tasks, FLOOR, use_advisor=False)

    assert decision.task_ids == FLOOR
    assert decision.advisor_status == "disabled"


def test_ids_the_suite_does_not_contain_are_not_smuggled_in_by_the_rules_either():
    """Impact analysis naming a stale task id must not create a phantom entry."""
    tasks = make_tasks()
    decision = triage(tasks, {"t0", "deleted-task"}, use_advisor=False)

    assert decision.task_ids == {"t0"}


def test_advisor_runs_at_temperature_zero():
    """The same diff must select the same tests.

    At the agent's configured temperature, two runs of one commit proposed
    different regressions -- which would make a triaged run unreproducible.
    """
    tasks = make_tasks()
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(content='{"tasks": [], "regressions": []}')

    with patch("app.agent.llm.get_llm", return_value=fake) as get_llm:
        triage_mod.ask_advisor(tasks, FLOOR, ["a.py"], ["r1"], {})

    assert get_llm.call_args.kwargs.get("temperature") == 0.0


def test_selected_tasks_preserves_suite_order():
    """Report ordering follows the suite, not the advisor's reply."""
    tasks = make_tasks()
    proposal = {"tasks": [{"id": "t4", "reason": "x"}, {"id": "t2", "reason": "y"}], "regressions": []}
    with patch.object(triage_mod, "ask_advisor", return_value=proposal):
        decision = triage(tasks, FLOOR)

    assert [t.id for t in selected_tasks(tasks, decision)] == ["t0", "t1", "t2", "t4"]


def test_a_fenced_json_reply_is_parsed():
    """Models wrap JSON in code fences regardless of instructions."""
    tasks = make_tasks()
    fake = MagicMock()
    fake.invoke.return_value = MagicMock(
        content='```json\n{"tasks": [{"id": "t2", "reason": "r"}], "regressions": []}\n```'
    )
    with patch("app.agent.llm.get_llm", return_value=fake):
        out = triage_mod.ask_advisor(tasks, FLOOR, ["a.py"], [], {})

    assert out["tasks"][0]["id"] == "t2"


def test_a_cached_decision_is_replayed_instead_of_asking_again(tmp_path):
    """The advisor is not deterministic, so a re-run of one commit must not re-ask.

    Three runs of an unchanged diff against Gemini at temperature 0 returned
    three different proposals. The floor held every time, but the reported test
    set moved, which is not explicable to a reviewer.
    """
    tasks = make_tasks()
    proposal = {"tasks": [{"id": "t3", "reason": "first answer"}], "regressions": []}

    with patch.object(triage_mod, "ask_advisor", return_value=proposal) as advisor:
        first = triage(tasks, FLOOR, changed_files=["a.py"],
                       cache_dir=str(tmp_path), suite_sha="sha-1")
    assert advisor.call_count == 1
    assert first.replayed is False

    # A second, different answer must not be reachable: the advisor is not called.
    other = {"tasks": [{"id": "t5", "reason": "second answer"}], "regressions": []}
    with patch.object(triage_mod, "ask_advisor", return_value=other) as advisor:
        second = triage(tasks, FLOOR, changed_files=["a.py"],
                        cache_dir=str(tmp_path), suite_sha="sha-1")

    assert advisor.call_count == 0
    assert second.task_ids == first.task_ids
    assert second.replayed is True
    assert second.advisor_status == "replayed"


def test_editing_the_suite_invalidates_a_cached_decision(tmp_path):
    tasks = make_tasks()
    proposal = {"tasks": [], "regressions": []}
    with patch.object(triage_mod, "ask_advisor", return_value=proposal):
        triage(tasks, FLOOR, changed_files=["a.py"], cache_dir=str(tmp_path), suite_sha="sha-1")

    with patch.object(triage_mod, "ask_advisor", return_value=proposal) as advisor:
        triage(tasks, FLOOR, changed_files=["a.py"], cache_dir=str(tmp_path), suite_sha="sha-2")

    assert advisor.call_count == 1


def test_a_cached_decision_is_not_replayed_across_models(tmp_path):
    """A proposal from one model must not be presented as another model's."""
    tasks = make_tasks()
    keys = set()
    for provider, model in (("google", "gemini-2.5-flash"), ("groq", "llama-3.3-70b-versatile")):
        with patch("app.config.settings.LLM_PROVIDER", provider), \
             patch("app.config.settings.MODEL_NAME", model):
            keys.add(triage_mod._proposal_key(["a.py"], FLOOR, "sha-1"))

    assert len(keys) == 2


def test_a_replayed_decision_is_revalidated_against_the_live_suite(tmp_path):
    """A task deleted since the entry was written must be rejected, not resurrected."""
    tasks = make_tasks()
    proposal = {"tasks": [{"id": "t5", "reason": "existed when cached"}], "regressions": []}
    with patch.object(triage_mod, "ask_advisor", return_value=proposal):
        triage(tasks, FLOOR, changed_files=["a.py"], cache_dir=str(tmp_path), suite_sha="sha-1")

    shrunk = make_tasks(n=5)  # t5 no longer exists
    with patch.object(triage_mod, "ask_advisor", side_effect=AssertionError("must not be called")):
        replayed = triage(shrunk, FLOOR, changed_files=["a.py"],
                          cache_dir=str(tmp_path), suite_sha="sha-1")

    assert "t5" not in replayed.task_ids
    assert any("t5" in r for r in replayed.rejected)


def test_a_cache_dir_without_a_suite_hash_is_refused():
    tasks = make_tasks()
    with pytest.raises(ValueError, match="suite_sha"):
        triage(tasks, FLOOR, cache_dir="somewhere")
