"""
Insight engine: turn a history of runs into statements worth reading.

A table of numbers makes a reader do the comparison themselves, and most of the
time they will not. This module does it: it diffs the latest run against a
reference, decides whether each change is real or noise, and writes the finding
in words.

The decision is the point. A pass-rate drop is only reported as a regression when
a one-tailed Fisher exact test says the change is unlikely to be sampling noise,
using the same test that gates the regression experiment. Everything else is
labelled as movement within noise, so a dashboard cannot manufacture alarm from a
single unlucky run.
"""

from typing import Any, Dict, List, Optional

from app.evaluation import statistics as stats

# Below this, a proportion change is not worth a card even when significant.
# Prevents a wall of trivia on a suite with many metrics.
MIN_REPORTABLE_DELTA = 0.02

# Relative change at which a cost or latency move is called out.
PERFORMANCE_DELTA_RATIO = 0.20

ALPHA = 0.05

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2, "good": 3}


def _card(severity: str, title: str, detail: str, metric: str = "",
          delta: Optional[float] = None) -> Dict[str, Any]:
    return {"severity": severity, "title": title, "detail": detail,
            "metric": metric, "delta": delta}


def _pass_counts(record: Dict[str, Any]) -> Optional[tuple]:
    """(passed, total) for a run, or None when it cannot be determined."""
    total = record.get("total_tasks_evaluated")
    passed = (record.get("performance") or {}).get("tasks_passed")
    if total and passed is not None:
        return passed, total
    return None


def compare_runs(current: Dict[str, Any],
                 reference: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Findings for `current`, measured against `reference`.

    `reference` is normally the previous run on the same branch, or the last run
    on main for a pull request. With no reference, only absolute conditions are
    reported -- there is nothing to diff against, and inventing a comparison
    would be worse than saying so.
    """
    cards: List[Dict[str, Any]] = []

    # --- absolute conditions, no reference needed ------------------------
    if not current.get("complete", True):
        cards.append(_card(
            "warning", "Run did not complete",
            current.get("budget_stop_reason")
            or "The run stopped before every task was evaluated, so these numbers "
               "describe part of the suite only.",
        ))

    if current.get("gate_passed") is False:
        cards.append(_card(
            "critical", "CI gate failed",
            "The build was blocked. See the run log for which thresholds were violated.",
        ))

    catch = (current.get("regression_catch_rate") or {}).get("overall")
    if catch is None:
        cards.append(_card(
            "warning", "Regression detection not measured",
            "No faults were injected, so this run provides no evidence that the "
            "suite can still detect a regression.",
        ))

    if (current.get("performance") or {}).get("token_source") == "estimated":
        cards.append(_card(
            "info", "Cost figures are estimated",
            "The provider reported no token usage, so tokens were inferred from "
            "character counts. Treat the cost as a shadow figure, not a measurement.",
        ))

    instrument = current.get("instrument") or {}
    if instrument.get("degrading_detection_rate") is not None:
        rate = instrument["degrading_detection_rate"]
        mde = instrument.get("minimum_detectable_effect")
        cards.append(_card(
            "good" if rate >= 1.0 else "warning",
            f"Suite detected {rate * 100:.0f}% of regressions that degraded the agent",
            f"{instrument.get('regressions_that_degraded', 0)} of "
            f"{instrument.get('regressions_planted', 0)} planted regressions actually "
            f"caused harm"
            + (f"; resolution is {mde:.3f} pass-rate points." if mde else "."),
        ))

    if reference is None:
        cards.append(_card(
            "info", "No earlier run to compare against",
            "Only absolute checks were applied. A comparison appears once a "
            "previous run exists for this branch.",
        ))
        return _sorted(cards)

    # --- pass rate, tested for significance -------------------------------
    current_counts, reference_counts = _pass_counts(current), _pass_counts(reference)
    if current_counts and reference_counts:
        cur_pass, cur_total = current_counts
        ref_pass, ref_total = reference_counts
        cur_rate, ref_rate = cur_pass / cur_total, ref_pass / ref_total
        delta = round(cur_rate - ref_rate, 4)

        if abs(delta) >= MIN_REPORTABLE_DELTA:
            if delta < 0:
                p_value = stats.fisher_exact_decrease(ref_pass, ref_total, cur_pass, cur_total)
                if p_value <= ALPHA:
                    cards.append(_card(
                        "critical", f"Pass rate dropped {abs(delta) * 100:.1f} points",
                        f"{ref_pass}/{ref_total} to {cur_pass}/{cur_total}. "
                        f"Fisher exact p={p_value:.4f}, so this is unlikely to be "
                        f"sampling noise.",
                        "pass_rate", delta,
                    ))
                else:
                    cards.append(_card(
                        "info", f"Pass rate moved down {abs(delta) * 100:.1f} points",
                        f"{ref_pass}/{ref_total} to {cur_pass}/{cur_total}, but "
                        f"Fisher exact p={p_value:.4f} does not clear {ALPHA}. "
                        f"Not distinguishable from noise at this many tasks.",
                        "pass_rate", delta,
                    ))
            else:
                p_value = stats.fisher_exact_decrease(cur_pass, cur_total, ref_pass, ref_total)
                cards.append(_card(
                    "good" if p_value <= ALPHA else "info",
                    f"Pass rate improved {delta * 100:.1f} points",
                    f"{ref_pass}/{ref_total} to {cur_pass}/{cur_total}, "
                    f"Fisher exact p={p_value:.4f}."
                    + ("" if p_value <= ALPHA else " Below significance; treat as noise."),
                    "pass_rate", delta,
                ))

    # --- per-metric movement ---------------------------------------------
    cur_metrics = current.get("summary_metrics") or {}
    ref_metrics = reference.get("summary_metrics") or {}
    for name in sorted(set(cur_metrics) & set(ref_metrics)):
        cur_value, ref_value = cur_metrics[name], ref_metrics[name]
        if cur_value is None or ref_value is None:
            continue
        delta = round(cur_value - ref_value, 4)
        if abs(delta) < MIN_REPORTABLE_DELTA:
            continue
        # Safety is the one metric where any drop is worth escalating: a task
        # that stopped refusing is not a tuning matter.
        critical = name == "safety_and_policy" and delta < 0
        cards.append(_card(
            "critical" if critical else ("warning" if delta < 0 else "good"),
            f"{name.replace('_', ' ').title()} {'fell' if delta < 0 else 'rose'} "
            f"{abs(delta):.3f}",
            f"{ref_value:.3f} to {cur_value:.3f}"
            + (" -- safety must never regress." if critical else "."),
            name, delta,
        ))

    # --- cost and latency, reported as ratios ------------------------------
    cur_perf = current.get("performance") or {}
    ref_perf = reference.get("performance") or {}
    for key, label, unit in (
        ("cost_per_successful_task_usd", "Cost per successful task", "$"),
        ("average_latency_seconds", "Average latency", "s"),
        ("average_total_tokens", "Average tokens per task", ""),
    ):
        cur_value, ref_value = cur_perf.get(key), ref_perf.get(key)
        if not cur_value or not ref_value:
            continue
        ratio = (cur_value - ref_value) / ref_value
        if abs(ratio) < PERFORMANCE_DELTA_RATIO:
            continue
        cards.append(_card(
            "warning" if ratio > 0 else "good",
            f"{label} {'up' if ratio > 0 else 'down'} {abs(ratio) * 100:.0f}%",
            f"{unit}{ref_value:,.5g} to {unit}{cur_value:,.5g}.",
            key, round(ratio, 4),
        ))

    return _sorted(cards)


def _sorted(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Most severe first, so the important finding is never below the fold."""
    return sorted(cards, key=lambda c: SEVERITY_ORDER.get(c["severity"], 9))


def find_reference(history: List[Dict[str, Any]],
                   current_index: int = -1) -> Optional[Dict[str, Any]]:
    """
    The run the latest one should be measured against.

    A pull request is compared with the most recent run on main, which is the
    state it would merge into. Anything else is compared with the run before it
    on the same branch.
    """
    if len(history) < 2:
        return None

    current = history[current_index]
    earlier = history[:current_index] if current_index != -1 else history[:-1]
    ci = current.get("ci") or {}

    if ci.get("pr_number"):
        main_runs = [
            r for r in earlier
            if (r.get("ci") or {}).get("branch") in ("main", "master")
            and not (r.get("ci") or {}).get("pr_number")
        ]
        if main_runs:
            return main_runs[-1]

    branch = ci.get("branch")
    same_branch = [r for r in earlier if (r.get("ci") or {}).get("branch") == branch]
    return same_branch[-1] if same_branch else earlier[-1]
