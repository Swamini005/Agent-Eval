"""
Token and cost budget for a run.

A free tier is a hard wall, and hitting it mid-suite is the worst place to find
out: some tasks have run, the rest fail with 429s, and the resulting report mixes
real measurements with rate-limit failures while looking like a complete run.

Measured against Groq's free tier on 2026-07-30: 100,000 tokens/day, and this
agent spends roughly 10,000 tokens on a single task because it re-sends a growing
message history on every loop. That is ten tasks per day. A 30-task suite cannot
complete, and nothing in the harness said so until the quota was gone.

The guard stops the run at a declared ceiling and says why, so a partial run is
labelled as partial instead of being read as a result.
"""

import threading
from typing import Optional


class BudgetExceeded(RuntimeError):
    """Raised when a run reaches its declared token or cost ceiling."""


class RunBudget:
    """
    Cumulative token and cost counter for one run.

    Thread-safe: the benchmark runner executes tasks concurrently, and an
    unsynchronised counter would undercount exactly when the ceiling matters.
    """

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        max_cost_usd: Optional[float] = None,
    ):
        self.max_tokens = max_tokens
        self.max_cost_usd = max_cost_usd
        self.tokens = 0
        self.cost_usd = 0.0
        self.tasks_charged = 0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_tokens is not None or self.max_cost_usd is not None

    def charge(self, tokens: int, cost_usd: float) -> None:
        """
        Record one task's usage.

        Raises:
            BudgetExceeded: the ceiling has been reached. Raised *after*
                recording, so the reported totals include the task that crossed
                the line rather than silently omitting it.
        """
        with self._lock:
            self.tokens += tokens
            self.cost_usd += cost_usd
            self.tasks_charged += 1

            if self.max_tokens is not None and self.tokens >= self.max_tokens:
                raise BudgetExceeded(
                    f"Token budget exhausted: {self.tokens:,} of {self.max_tokens:,} "
                    f"after {self.tasks_charged} tasks "
                    f"(~{self.tokens // max(1, self.tasks_charged):,} per task). "
                    f"Raise --max-tokens, use a cheaper model, or run a smaller suite."
                )
            if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
                raise BudgetExceeded(
                    f"Cost budget exhausted: ${self.cost_usd:.4f} of "
                    f"${self.max_cost_usd:.4f} after {self.tasks_charged} tasks."
                )

    def remaining_tasks(self) -> Optional[int]:
        """Rough estimate of how many more tasks fit, from the average so far."""
        if self.max_tokens is None or not self.tasks_charged:
            return None
        per_task = self.tokens / self.tasks_charged
        if per_task <= 0:
            return None
        return max(0, int((self.max_tokens - self.tokens) / per_task))

    def summary(self) -> dict:
        return {
            "tokens_used": self.tokens,
            "max_tokens": self.max_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "max_cost_usd": self.max_cost_usd,
            "tasks_charged": self.tasks_charged,
            "tokens_per_task": (
                round(self.tokens / self.tasks_charged) if self.tasks_charged else 0
            ),
            "estimated_tasks_remaining": self.remaining_tasks(),
        }
