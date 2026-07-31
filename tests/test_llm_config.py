"""Model resolution, and the line between deterministic mode and a real failure.

Every LLM call site used to catch bare `Exception` and substitute a rule-based
answer, so a run whose model was completely broken scored the same as a healthy
one. These tests hold that line.
"""

import pytest

from app.agent.llm import DeterministicMode, get_llm, token_usage
from app.agent.nodes import intent_detection_node
from app.agent.state import AgentState
from app.config import Settings
from langchain_core.messages import HumanMessage


def state(text="Find a flight to Paris"):
    return {"messages": [HumanMessage(content=text)], "intent": {}, "plan": [],
            "current_step": 0, "tool_calls": [], "tool_results": [],
            "reasoning": "", "response": ""}


def test_mock_provider_raises_deterministic_mode_not_a_generic_error():
    """Deterministic mode is a configuration, not a failure, and is typed as one."""
    with pytest.raises(DeterministicMode):
        get_llm()


def test_deterministic_mode_still_produces_a_rule_based_answer():
    result = intent_detection_node(state())
    assert result["intent"]["flights"] is True


def test_a_real_model_error_is_not_swallowed(monkeypatch):
    """The point of the change: a broken model must fail the task, not be hidden.

    A rate limit or bad key previously fell through to the rule-based path and
    the suite reported a healthy score for an agent that never reached its model.
    """
    def exploding_llm():
        raise RuntimeError("429 rate limit exceeded")

    monkeypatch.setattr("app.agent.nodes.get_llm", exploding_llm)

    with pytest.raises(RuntimeError, match="429"):
        intent_detection_node(state())


def test_unknown_provider_is_a_configuration_error(monkeypatch):
    """A typo in the provider name must not look like a clean deterministic run."""
    monkeypatch.setattr("app.agent.llm.settings", Settings(LLM_PROVIDER="gorq"))
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        get_llm()


def test_missing_key_is_reported_with_the_variable_to_set(monkeypatch):
    monkeypatch.setattr("app.agent.llm.settings", Settings(LLM_PROVIDER="groq", GROQ_API_KEY=None))
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        get_llm()


def test_openai_compatible_providers_resolve_a_base_url(monkeypatch):
    """Groq and friends are reachable without a code change."""
    from app.agent import llm as llm_module

    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_module, "settings",
                        Settings(LLM_PROVIDER="groq", GROQ_API_KEY="test-key"))
    monkeypatch.setitem(__import__("sys").modules, "langchain_openai",
                        type("m", (), {"ChatOpenAI": FakeChat}))

    get_llm()

    assert captured["base_url"] == "https://api.groq.com/openai/v1"
    assert captured["api_key"] == "test-key"
    assert captured["temperature"] == 0.0
    # A real model needs bounds, or one hung call stalls the whole run.
    assert captured["timeout"] > 0
    assert captured["max_retries"] >= 0


def test_token_usage_distinguishes_unreported_from_zero():
    """Estimating and presenting the estimate as measured is what this avoids."""
    assert token_usage(object()) == {}

    reported = type("R", (), {"usage_metadata": {"input_tokens": 12, "output_tokens": 5}})()
    assert token_usage(reported) == {"prompt_tokens": 12, "completion_tokens": 5}


def test_importing_the_agent_installs_no_global_llm_cache():
    """A process-global LLM cache would silently destroy the seed measurement.

    `app/agent/nodes.py` called set_llm_cache(InMemoryCache()) at import. Seeds
    exist to sample a stochastic agent, and for the clean arm the prompts are
    identical across seeds -- so every seed after the first replayed the first
    one's response. Five seeds became one sample and four copies: stdev was 0 by
    construction and the confidence intervals described replicas, not runs.

    It was global, so it also applied to third-party agents under test and to the
    triage advisor, neither of which imported it. Caching belongs at the run
    level, where RegressionExperiment keys it on seed, arm, provider and suite.
    """
    from langchain_core.globals import get_llm_cache

    import app.agent.nodes  # noqa: F401  -- import for its side effects, if any

    assert get_llm_cache() is None


def test_the_test_suite_is_hermetic_regardless_of_any_local_env_file():
    """A developer .env must not turn the unit tests into live model calls.

    The deepeval pytest plugin loads .env into os.environ before conftest runs,
    so conftest's setdefault() silently lost to it. Adding a .env with a real
    provider took this file from 0.6s to 32s, spent tokens on every run, and
    failed outright when the key was missing -- a failure that looks like a code
    defect and is not one.
    """
    import os

    from app.agent.llm import is_deterministic
    from app.config import settings

    if os.environ.get("EVAL_ALLOW_LIVE_TESTS") == "1":
        return  # deliberately opted into a live run

    assert settings.LLM_PROVIDER == "mock"
    assert is_deterministic() is True
