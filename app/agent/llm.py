"""
Model resolution for the agent under test.

Two distinct situations were previously conflated, and separating them is the
point of this module.

`DeterministicMode` is a *deliberate* configuration: LLM_PROVIDER=mock means the
agent must take its rule-based path, so CI and the test suite run with no API key,
no network and a reproducible result. That is not a failure.

A real LLM error -- a bad key, a rate limit, a malformed response -- *is* a
failure, and it propagates. Previously every node caught `Exception` and quietly
substituted the rule-based path, so a run whose model was entirely broken scored
the same as a healthy one. The suite would have reported a green result for an
agent that never reached its model, which is precisely the blind spot this
project exists to close.
"""

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings

# OpenAI-compatible endpoints, selected by provider name. Groq is the default
# free-tier target; any other compatible endpoint works via LLM_BASE_URL.
OPENAI_COMPATIBLE_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}

DEFAULT_MODELS = {
    "google": "gemini-2.5-flash-lite",
    "openai": "gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
}


class DeterministicMode(Exception):
    """
    Raised when the agent is configured to run without a model.

    Nodes catch this specific type -- and nothing else -- to select their
    rule-based path. Catching broad exceptions here is what previously let a
    genuine model failure masquerade as deterministic mode.
    """


def is_deterministic() -> bool:
    return settings.LLM_PROVIDER.lower() == "mock"


def get_llm(temperature: Optional[float] = None) -> BaseChatModel:
    """
    Build the configured chat model.

    `temperature` overrides the configured value. Used by callers whose output
    feeds a decision rather than an answer -- triage passes 0.0, because the set
    of tests a commit runs must not vary between two runs of the same diff.

    Raises:
        DeterministicMode: LLM_PROVIDER=mock; the caller should use its
            rule-based path.
        ValueError: the provider is unrecognised, or its key is missing. Both
            are configuration errors and must not silently degrade into
            deterministic mode -- a typo in the provider name would otherwise
            look like a clean rule-based run.
    """
    provider = settings.LLM_PROVIDER.lower()

    if provider == "mock":
        raise DeterministicMode("LLM_PROVIDER=mock: using the rule-based path.")

    temperature = settings.TEMPERATURE if temperature is None else temperature
    model_name = settings.MODEL_NAME or DEFAULT_MODELS.get(provider)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not settings.GOOGLE_API_KEY:
            raise ValueError("LLM_PROVIDER=google but GOOGLE_API_KEY is not set.")
        # Bounded like the OpenAI-compatible branch. Without a timeout a flaky
        # connection stalls the whole run: a dropped socket produced no error
        # and no progress, because nothing ever gave up on the request.
        # Passed explicitly rather than left to the library's os.environ lookup.
        # Settings also read .env, so a key configured only there satisfied the
        # check above and then failed to authenticate -- the one arrangement most
        # likely to be used, and the failure looked like a bad key.
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=settings.GOOGLE_API_KEY,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    if provider == "openai" or provider in OPENAI_COMPATIBLE_BASE_URLS:
        from langchain_openai import ChatOpenAI

        api_key = settings.api_key_for(provider)
        if not api_key:
            raise ValueError(
                f"LLM_PROVIDER={provider} but no API key is set. "
                f"Expected {settings.key_env_var(provider)}."
            )
        base_url = settings.LLM_BASE_URL or OPENAI_COMPATIBLE_BASE_URLS.get(provider)
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER={provider!r}. "
        f"Expected 'mock', 'google', 'openai', or one of "
        f"{sorted(OPENAI_COMPATIBLE_BASE_URLS)}."
    )


def text_of(response) -> str:
    """
    The text of a model response, whatever shape the provider returned.

    OpenAI-compatible providers set `content` to a string. Gemini 3.x sets it to
    a list of typed content blocks:

        [{"type": "text", "text": "...", "extras": {...}}]

    Calling .strip() on that raises AttributeError, so every node that parsed a
    response worked against Groq and OpenAI and failed against Gemini. Normalising
    here keeps provider shape out of the agent.
    """
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def token_usage(response) -> dict:
    """
    Real prompt/completion counts from a model response.

    Returns an empty dict when the provider reports nothing, so callers can tell
    "no usage reported" from "zero tokens" instead of estimating and presenting
    the estimate as measured.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        }

    metadata = getattr(response, "response_metadata", None) or {}
    raw = metadata.get("token_usage") or metadata.get("usage") or {}
    if raw:
        return {
            "prompt_tokens": raw.get("prompt_tokens", 0),
            "completion_tokens": raw.get("completion_tokens", 0),
        }
    return {}
