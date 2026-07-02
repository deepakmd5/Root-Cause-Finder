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

    # --- Database (PostgreSQL, optional) ---
    # Empty string disables DB integration. When set, the query_database
    # tool will use it. Example:
    #   postgresql://rca_ro:secret@localhost:5432/rca_ops
    database_url: str = ""
    database_pool_min: int = Field(default=1, ge=0, le=50)
    database_pool_max: int = Field(default=5, ge=1, le=100)
    database_query_timeout_seconds: float = Field(default=5.0, ge=0.1, le=60.0)
    database_statement_timeout_ms: int = Field(default=5000, ge=100, le=60000)

    # --- Aerospike (NoSQL, optional) ---
    # Empty AEROSPIKE_HOSTS disables integration. When both hosts +
    # namespace are set, the query_aerospike tool becomes available.
    # Hosts format: comma-separated ``host:port`` list, e.g.
    #   "cache-1.internal:3000,cache-2.internal:3000"
    aerospike_hosts: str = ""
    aerospike_namespace: str = ""
    aerospike_username: str = ""
    aerospike_password: str = ""
    aerospike_total_timeout_ms: int = Field(default=1000, ge=50, le=60000)
    aerospike_query_timeout_seconds: float = Field(
        default=2.0, ge=0.1, le=60.0
    )


@lru_cache
def get_settings() -> Settings:
    """Return a memoized Settings instance."""
    return Settings()
