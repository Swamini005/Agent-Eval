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
    models: Dict[str, ModelRate] = Field(default_factory=dict)
    families: Dict[str, ModelRate] = Field(default_factory=dict)

    def rate_for(self, model: str) -> Optional[ModelRate]:
        """
        Rate for a model: exact name first, then the longest matching prefix.

        The prefix layer stops a new point release from silently costing zero.
        `gemini-3.5-flash-lite-preview-0409` has no exact entry but matches the
        `gemini-3.5-flash-lite` family. Longest match wins, so
        `gemini-2.5-flash-lite` is never priced as `gemini-2.5-flash`.
        """
        if model in self.models:
            return self.models[model]

        matches = [name for name in self.families if model.startswith(name)]
        if not matches:
            return None
        return self.families[max(matches, key=len)]

    def cost(self, prompt_tokens: int, completion_tokens: int,
             model: Optional[str] = None) -> float:
        """
        Cost of one call, in USD, at the rates for the model actually in use.

        An unknown model is priced at zero rather than guessed. Inventing a rate
        produces a confident number with no basis, which is worse than a visible
        zero -- and pricing a Groq run at Gemini rates, as this did before the
        model was resolved from configuration, is exactly that failure.
        """
        rate = self.rate_for(model or active_model())
        if rate is None:
            # Zero, but the caller must be able to tell this apart from a model
            # that genuinely costs nothing. See has_rate(): reporting an unpriced
            # model as $0.00 reads as "free" and is the same class of error as
            # reporting an unmeasured metric as a pass.
            return 0.0
        return round(
            (prompt_tokens / 1_000_000) * rate.prompt_per_1m
            + (completion_tokens / 1_000_000) * rate.completion_per_1m,
            8,
        )


def has_rate(model: Optional[str] = None) -> bool:
    """Whether a rate exists for this model, exactly or by family prefix."""
    return load_pricing().rate_for(model or active_model()) is not None


def active_model() -> str:
    """
    The model this run is using, resolved the same way the agent resolves it.

    Falls back to the price list's default only in deterministic mode, where no
    model is involved and the figure is a shadow cost.
    """
    from app.agent.llm import DEFAULT_MODELS
    from app.config import settings

    if settings.MODEL_NAME:
        return settings.MODEL_NAME
    provider = settings.LLM_PROVIDER.lower()
    if provider in DEFAULT_MODELS:
        return DEFAULT_MODELS[provider]
    return load_pricing().default_model


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
