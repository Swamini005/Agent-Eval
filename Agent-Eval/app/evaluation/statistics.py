"""
Statistics for gating a stochastic system deterministically.

An agent is not a pure function: the same task can pass on one run and fail on
the next. A gate that compares a single run's mean against a threshold therefore
flakes, and a flaky merge blocker is worse than no blocker at all -- people learn
to re-run it until it goes green.

The approach here is to treat each task as a Bernoulli trial repeated across
seeds, report a pass *rate* rather than a boolean, and gate on the lower bound of
a confidence interval. A build then fails only when the evidence for a regression
outweighs the observed noise.
"""

import math
from typing import Dict, List, Optional, Sequence

# 95% two-sided normal quantile. Fixed rather than configurable: a gate whose
# confidence level can be tuned per run is a gate that can be argued down.
Z_95 = 1.959963984540054


def wilson_interval(successes: int, trials: int, z: float = Z_95) -> Dict[str, float]:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because agent suites run few seeds
    and often sit near 0 or 1, where the normal interval produces bounds outside
    [0, 1] and badly understates uncertainty.
    """
    if trials <= 0:
        return {"point": 0.0, "lower": 0.0, "upper": 1.0}

    p_hat = successes / trials
    denominator = 1 + (z ** 2) / trials
    center = (p_hat + (z ** 2) / (2 * trials)) / denominator
    margin = z * math.sqrt(
        (p_hat * (1 - p_hat) / trials) + (z ** 2) / (4 * trials ** 2)
    ) / denominator

    return {
        "point": round(p_hat, 4),
        "lower": round(max(0.0, center - margin), 4),
        "upper": round(min(1.0, center + margin), 4),
    }


def pass_rate(outcomes: Sequence[bool]) -> Dict[str, float]:
    """Pass rate with its confidence interval, from per-seed outcomes."""
    successes = sum(1 for outcome in outcomes if outcome)
    stats = wilson_interval(successes, len(outcomes))
    stats.update({"passed": successes, "trials": len(outcomes)})
    return stats


def is_flaky(outcomes: Sequence[bool]) -> bool:
    """
    True when a task neither reliably passes nor reliably fails.

    Flaky tasks carry no signal about a regression -- they change verdict on
    their own -- so they are quarantined and reported rather than being allowed
    to move the headline number.
    """
    return len(set(bool(o) for o in outcomes)) > 1


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stdev(values: Sequence[float]) -> float:
    """Sample standard deviation; 0.0 for fewer than two observations."""
    if len(values) < 2:
        return 0.0
    average = mean(values)
    variance = sum((v - average) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def fisher_exact_decrease(
    clean_pass: int, clean_total: int, faulted_pass: int, faulted_total: int
) -> float:
    """
    One-tailed Fisher exact p-value for "the faulted arm passed less often".

    Used instead of checking whether two confidence intervals overlap. Interval
    overlap is a much stricter test than a direct comparison, and at the seed
    counts an agent suite can afford it never fires: at three seeds the Wilson
    intervals for 3/3 and 0/3 still overlap, so a task that flips from always
    passing to always failing would be reported as undetected.

    Fisher is exact for small counts, which is precisely the regime here.
    """
    a, b = faulted_pass, faulted_total - faulted_pass
    c, d = clean_pass, clean_total - clean_pass
    n = a + b + c + d
    if n == 0:
        return 1.0

    row1, row2 = a + b, c + d
    col1 = a + c

    def table_probability(x: int) -> float:
        y = row1 - x
        z = col1 - x
        w = row2 - z
        if min(y, z, w) < 0:
            return 0.0
        return (
            math.comb(row1, x) * math.comb(row2, z) / math.comb(n, col1)
        )

    # Sum the probability of the observed table and every table with fewer
    # passes in the faulted arm.
    return round(sum(table_probability(x) for x in range(0, a + 1)), 6)


def minimum_detectable_effect(
    detections: List[Dict[str, float]],
    required_rate: float = 1.0,
) -> Optional[float]:
    """
    Smallest true degradation the suite reliably detects -- its resolution.

    Takes one entry per planted regression as {"effect": <true drop>,
    "detection_rate": <fraction of seeds where the gate fired>} and returns the
    smallest effect whose detection rate met `required_rate`, provided every
    larger effect was also detected. Returns None when nothing was reliably
    detected.

    Reporting "100% of regressions detected" is only meaningful alongside this
    number: without it the claim silently omits the effect sizes that were never
    planted and therefore never caught.
    """
    if not detections:
        return None

    ordered = sorted(detections, key=lambda d: d["effect"], reverse=True)
    resolution = None
    for entry in ordered:
        if entry["detection_rate"] >= required_rate:
            resolution = entry["effect"]
        else:
            # Once a larger effect is missed, smaller ones detected below it are
            # coincidence rather than resolution.
            break
    return round(resolution, 4) if resolution is not None else None
