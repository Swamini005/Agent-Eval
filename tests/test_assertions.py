"""Ground-truth assertions: the only metric that yields a definitive verdict."""

from app.evaluation.metrics.assertion import AssertionMetric
from app.evaluation.models import EvaluationTaskInput, EvaluationExecutionInput


def make(ground_truth, tool_calls=(), response="", latency=0.1, domain="travel"):
    task = EvaluationTaskInput(task_id="t1", benchmark="b", category="c", domain=domain,
                               prompt="p", ground_truth=ground_truth)
    execution = EvaluationExecutionInput(task_id="t1", response=response, latency_seconds=latency,
                                         cost_usd=0.0, tool_calls=list(tool_calls))
    return AssertionMetric().evaluate(task, execution, {"injections": []})


def test_no_assertions_is_unmeasured_not_a_free_pass():
    result = make(None)
    assert result.measured is False
    assert result.score == 0.0


def test_unrecognised_clause_is_surfaced_not_silently_ignored():
    """A typo in a task file must not quietly reduce that task's coverage to zero."""
    result = make({"must_call_toolz": ["search_flights"]})
    assert result.measured is False
    assert result.details["unknown_clauses"] == ["must_call_toolz"]


def test_required_and_forbidden_tools():
    calls = [{"tool_name": "search_flights", "args": {}}]
    assert make({"must_call_tools": ["search_flights"]}, calls).details["all_passed"] is True
    assert make({"must_call_tools": ["book_flight"]}, calls).details["all_passed"] is False
    assert make({"must_not_call_tools": ["book_flight"]}, calls).details["all_passed"] is True
    assert make({"must_not_call_tools": ["search_flights"]}, calls).details["all_passed"] is False


def test_tool_order_allows_gaps_but_not_inversions():
    calls = [{"tool_name": t, "args": {}} for t in ("search_flights", "get_weather", "book_flight")]
    assert make({"must_call_tools_in_order": ["search_flights", "book_flight"]}, calls).score == 1.0
    assert make({"must_call_tools_in_order": ["book_flight", "search_flights"]}, calls).score == 0.0


def test_destructive_check_reads_the_domain_pack():
    booking = [{"tool_name": "book_flight", "args": {"price_usd": 1.0}}]
    assert make({"must_not_call_destructive": True}, booking).details["all_passed"] is False
    searching = [{"tool_name": "search_flights", "args": {}}]
    assert make({"must_not_call_destructive": True}, searching).details["all_passed"] is True


def test_destructive_check_cannot_pass_without_a_pack():
    """An undeclared domain must fail the assertion rather than pass it vacuously."""
    booking = [{"tool_name": "book_flight", "args": {}}]
    result = make({"must_not_call_destructive": True}, booking, domain="unknown-domain")
    assert result.details["all_passed"] is False


def test_response_content_and_budgets():
    assert make({"response_must_contain": ["TX-101"]}, response="Flight TX-101 found").score == 1.0
    assert make({"response_must_not_contain": ["$50"]}, response="the fee is $50").score == 0.0
    assert make({"max_tool_calls": 1}, [{"tool_name": "a"}, {"tool_name": "b"}]).score == 0.0
    assert make({"max_latency_seconds": 0.05}, latency=0.2).score == 0.0


def test_score_is_the_fraction_of_assertions_that_held():
    calls = [{"tool_name": "search_flights", "args": {}}]
    result = make({"must_call_tools": ["search_flights"], "must_not_call_tools": ["search_flights"]}, calls)
    assert result.score == 0.5
    assert result.details["all_passed"] is False
    assert len(result.details["failures"]) == 1


# --- trajectory matching, delegated to agentevals ---------------------------

def calls(*names):
    return [{"tool_name": n, "args": {}} for n in names]


def test_trajectory_superset_allows_extra_tools():
    gt = {"trajectory_match": {"mode": "superset", "reference_tools": ["search_flights"]}}
    assert make(gt, calls("search_flights", "get_weather")).details["all_passed"] is True
    assert make(gt, calls("get_weather")).details["all_passed"] is False


def test_trajectory_subset_forbids_extra_tools():
    gt = {"trajectory_match": {"mode": "subset",
                               "reference_tools": ["search_flights", "get_weather"]}}
    assert make(gt, calls("search_flights")).details["all_passed"] is True
    assert make(gt, calls("search_flights", "book_flight")).details["all_passed"] is False


def test_trajectory_strict_enforces_order():
    gt = {"trajectory_match": {"mode": "strict",
                               "reference_tools": ["search_flights", "convert_currency"]}}
    assert make(gt, calls("search_flights", "convert_currency")).details["all_passed"] is True
    assert make(gt, calls("convert_currency", "search_flights")).details["all_passed"] is False


def test_trajectory_unordered_ignores_order():
    gt = {"trajectory_match": {"mode": "unordered",
                               "reference_tools": ["search_flights", "convert_currency"]}}
    assert make(gt, calls("convert_currency", "search_flights")).details["all_passed"] is True


def test_unknown_trajectory_mode_fails_the_assertion():
    """An unrecognised mode must not silently fall back to a laxer comparison."""
    gt = {"trajectory_match": {"mode": "approximately", "reference_tools": ["search_flights"]}}
    result = make(gt, calls("search_flights"))

    assert result.details["all_passed"] is False
    assert "could not evaluate" in result.details["failures"][0]["detail"]
