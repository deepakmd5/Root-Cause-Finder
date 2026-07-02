"""Application configuration loaded from environment variables."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Values are read from environment variables (or a local ``.env`` file).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = Field(default="agentic-rca-finder")
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Agent ---
    agent_max_iterations: int = Field(default=8, ge=1, le=25)
    agent_timeout_seconds: int = Field(default=60, ge=5, le=600)

    # --- LLM ---
    llm_provider: Literal["mock", "openai"] = "mock"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1500


@lru_cache
def get_settings() -> Settings:
    """Return a memoized Settings instance."""
    return Settings()
