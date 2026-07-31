import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Every generated report goes here rather than the repository root, so a run
# never litters the working tree and a single .gitignore entry covers all of it.
# These files are outputs, not source: they are rebuilt by each run and must
# never be committed.
REPORTS_DIR = "reports"

# Which environment variable holds the key for each provider.
PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # LLM configuration.
    #
    # 'mock' runs the agent's deterministic rule-based path with no model, no key
    # and no network. It is the default so a fresh clone, CI and the test suite
    # all work unconfigured and reproducibly; set a real provider for a demo.
    #
    # Previously defaulted to 'google', which meant an unconfigured run attempted
    # a network call at every node and fell back after it failed -- slow, and it
    # made latency measurements meaningless.
    LLM_PROVIDER: str = "mock"
    MODEL_NAME: Optional[str] = None
    TEMPERATURE: float = 0.0

    # Overrides the provider's default endpoint, so any OpenAI-compatible
    # service works without a code change.
    LLM_BASE_URL: Optional[str] = None

    # A real model needs bounds. Without a timeout one hung request stalls the
    # whole run; without a retry cap a rate-limited key retries indefinitely.
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_RETRIES: int = 2

    # API keys
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    TOGETHER_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None

    # App Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    def key_env_var(self, provider: str) -> str:
        return PROVIDER_KEY_ENV.get(provider.lower(), "OPENAI_API_KEY")

    def api_key_for(self, provider: str) -> Optional[str]:
        """The configured key for a provider, or None."""
        return getattr(self, self.key_env_var(provider), None)


settings = Settings()


def _export_keys_to_environ() -> None:
    """
    Publish configured API keys into os.environ.

    pydantic-settings reads .env into this object without touching os.environ.
    An agent handed over for evaluation is not routed through app/agent/llm.py --
    it builds its own client, which reads os.environ directly -- so a key that
    lived only in .env was invisible to it and every task failed with "API key
    required". The suite then scored the agent on those failures, reporting a
    configuration problem as an agent defect.

    Existing environment variables win: a value exported by the shell or by CI is
    deliberate and must not be overridden by a file on disk.
    """
    for env_var in set(PROVIDER_KEY_ENV.values()):
        value = getattr(settings, env_var, None)
        if value and not os.environ.get(env_var):
            os.environ[env_var] = value

    # langchain-google-genai accepts either name.
    if settings.GOOGLE_API_KEY and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = settings.GOOGLE_API_KEY


_export_keys_to_environ()
