"""Statistics that let a deterministic gate sit in front of a stochastic system."""

from app.evaluation import statistics as stats


def test_wilson_bounds_stay_inside_zero_one_at_the_extremes():
    """The normal approximation produces bounds outside [0,1] here; Wilson does not."""
    for successes, trials in ((0, 3), (3, 3), (0, 1), (1, 1)):
        interval = stats.wilson_interval(successes, trials)
        assert 0.0 <= interval["lower"] <= interval["upper"] <= 1.0


def test_wilson_interval_narrows_as_seeds_increase():
    narrow = stats.wilson_interval(10, 10)
    wide = stats.wilson_interval(3, 3)
    assert narrow["lower"] > wide["lower"]


def test_no_trials_is_maximally_uncertain_not_confidently_zero():
    interval = stats.wilson_interval(0, 0)
    assert interval["lower"] == 0.0 and interval["upper"] == 1.0


def test_flaky_detection():
    assert stats.is_flaky([True, False, True]) is True
    assert stats.is_flaky([True, True, True]) is False
    assert stats.is_flaky([False, False]) is False


def test_seed_count_sets_the_evidence_available_for_a_full_flip():
    """Why the suite defaults to five seeds rather than three.

    A task that always passed and now always fails is the clearest regression
    there is, and the seed count alone caps how strong that evidence can be. At
    three seeds it lands exactly on the alpha boundary -- it qualifies, with no
    margin whatsoever, so a single flaky seed would push it back out.
    """
    assert stats.fisher_exact_decrease(3, 3, 0, 3) == 0.05
    assert stats.fisher_exact_decrease(4, 4, 0, 4) < 0.02
    assert stats.fisher_exact_decrease(5, 5, 0, 5) < 0.005

    # Monotone: more seeds, stronger evidence for the same effect.
    p_values = [stats.fisher_exact_decrease(n, n, 0, n) for n in (3, 4, 5, 6)]
    assert p_values == sorted(p_values, reverse=True)


def test_no_change_is_not_significant():
    assert stats.fisher_exact_decrease(5, 5, 5, 5) == 1.0


def test_minimum_detectable_effect_is_the_smallest_reliably_caught():
    detections = [
        {"effect": 0.40, "detection_rate": 1.0},
        {"effect": 0.23, "detection_rate": 1.0},
        {"effect": 0.05, "detection_rate": 0.0},
    ]
    assert stats.minimum_detectable_effect(detections) == 0.23


def test_minimum_detectable_effect_is_none_when_nothing_was_caught():
    assert stats.minimum_detectable_effect([{"effect": 0.4, "detection_rate": 0.0}]) is None
    assert stats.minimum_detectable_effect([]) is None


def test_stdev_of_a_deterministic_suite_is_zero():
    assert stats.stdev([0.867, 0.867, 0.867]) == 0.0
    assert stats.stdev([0.9]) == 0.0
