"""Task suites are data with a content hash, so a report can name what it measured."""



import pytest

from app.benchmarks.suites import compute_sha, load_suite, available_suites


def test_dev_and_holdout_suites_exist_and_are_separate():
    assert set(available_suites()) >= {"dev", "holdout"}

    dev, holdout = load_suite("dev"), load_suite("holdout")
    assert len(dev) >= 30, "the brief requires at least 30 tasks"
    assert len(holdout) >= 10
    assert dev.sha != holdout.sha
    assert not ({t.id for t in dev.tasks} & {t.id for t in holdout.tasks}), \
        "holdout must not overlap dev, or it cannot detect overfitting"


def test_every_task_declares_machine_checkable_assertions():
    """A prompt without a checker is not a test case."""
    for name in ("dev", "holdout"):
        for task in load_suite(name).tasks:
            assert task.ground_truth, f"{name}/{task.id} has no ground_truth"
            assert task.domain, f"{name}/{task.id} has no domain"


def test_dev_suite_covers_the_safety_critical_categories():
    categories = load_suite("dev").categories()
    for required in ("safety_gate", "context_corruption", "adversarial", "multi_step"):
        assert categories.get(required, 0) >= 5, f"thin coverage for {required}"


def test_hash_ignores_formatting_but_not_content():
    base = [{"id": "a", "prompt": "x", "ground_truth": {"max_tool_calls": 1}}]
    reordered = [{"prompt": "x", "ground_truth": {"max_tool_calls": 1}, "id": "a"}]
    changed = [{"id": "a", "prompt": "y", "ground_truth": {"max_tool_calls": 1}}]

    assert compute_sha(base) == compute_sha(reordered)
    assert compute_sha(base) != compute_sha(changed)


def test_hash_is_stable_across_loads():
    assert load_suite("dev").sha == load_suite("dev").sha


def test_missing_suite_fails_loudly():
    with pytest.raises(FileNotFoundError):
        load_suite("no-such-suite")


def test_reports_are_never_written_to_the_working_directory_by_default():
    """A caller that forgets output_dir must fail, not silently clobber the
    workspace reports the CI gate reads.

    Before this was enforced, running pytest overwrote evaluation_summary.json
    with three-task fixture data, so a subsequent gate_check graded the fixtures.
    """
    import inspect
    from app.evaluation.engine import EvaluationEngine

    signature = inspect.signature(EvaluationEngine.evaluate_run)
    assert signature.parameters["output_dir"].default is inspect.Parameter.empty
