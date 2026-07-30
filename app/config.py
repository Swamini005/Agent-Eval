from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

# Every generated report goes here rather than the repository root, so a run
# never litters the working tree and a single .gitignore entry covers all of it.
# These files are outputs, not source: they are rebuilt by each run and must
# never be committed.
REPORTS_DIR = "reports"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # LLM configurations
    # Can be 'google' or 'openai' or 'mock'
    LLM_PROVIDER: str = "google"
    MODEL_NAME: Optional[str] = None
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    
    # App Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

settings = Settings()
