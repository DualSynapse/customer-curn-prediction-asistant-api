from pydantic import Field
from typing import Literal
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore", 
        env_file_encoding="utf-8",
        protected_namespaces=("settings_",)
    )

    llm_provider: Literal["OpenAI", "anthropic"] = "OpenAI"

    model_name: str = "gemini-3.1-flash-lite-preview"
    temperature: float = 0.7
    max_tokens: int = 4096

    api_prefix: str = "/api/v1"
    cors_origins: list[str] = ["*"]

    GEMINI_API_KEY: str = Field(
        description="API key for authenticating with OpenAI services",
    )

    GEMINI_AGENT_LLM_MODEL: str = Field(
        default="gemini-3.1-flash-lite-preview",
        description="Primary Gemini model for general LLM task"
    )

@lru_cache
def get_settings()->Settings:
    """Get cached settings instance."""
    return Settings()