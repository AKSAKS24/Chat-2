from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    DEFAULT_MODELS: dict = {
        "openai": ["gpt-4o-mini", "gpt-4.1", "gpt-5.1"],
        "claude": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
    }

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
