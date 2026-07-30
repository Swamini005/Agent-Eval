"""
Token pricing, loaded from pricing.yaml.

Rates were previously inlined in the benchmark runner as two unnamed floats, so
cost figures could not be traced to any published price list and went stale
silently. Loading them from a versioned file means a report can state which
rates produced its numbers.
"""

import os
import threading
from typing import Dict, Optional

import yaml
from pydantic import BaseModel, Field

PRICING_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pricing.yaml"
)


class ModelRate(BaseModel):
    prompt_per_1m: float = Field(..., description="USD per 1M prompt tokens")
    completion_per_1m: float = Field(..., description="USD per 1M completion tokens")


class Pricing(BaseModel):
    default_model: str
    models: Dict[str, ModelRate]

    def cost(self, prompt_tokens: int, completion_tokens: int,
             model: Optional[str] = None) -> float:
        """
        Cost of one call, in USD.

        An unknown model is priced at zero rather than guessed. Inventing a rate
        would produce a confident number with no basis, which is worse than a
        visible zero.
        """
        rate = self.models.get(model or self.default_model)
        if rate is None:
            return 0.0
        return round(
            (prompt_tokens / 1_000_000) * rate.prompt_per_1m
            + (completion_tokens / 1_000_000) * rate.completion_per_1m,
            8,
        )


_cache: Optional[Pricing] = None
_lock = threading.Lock()


def load_pricing(path: str = PRICING_FILE) -> Pricing:
    """Load and cache the price list. Read from worker threads, so guarded."""
    global _cache
    with _lock:
        if _cache is None:
            with open(path, "r", encoding="utf-8") as f:
                _cache = Pricing(**yaml.safe_load(f))
        return _cache
