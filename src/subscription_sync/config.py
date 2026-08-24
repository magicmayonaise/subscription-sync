"""Type-safe configuration using Pydantic Settings.

Demonstrates:
- Environment variable loading with validation
- SecretStr for credential masking in logs
- Sensible defaults with override capability
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Notion
    notion_api_key: SecretStr
    notion_database_id: str
    notion_category_entertainment_page_id: str = ""
    notion_category_business_page_id: str = ""
    notion_category_education_page_id: str = ""
    notion_category_health_page_id: str = ""
    notion_category_decoration_page_id: str = ""

    # Anthropic
    anthropic_api_key: SecretStr

    # Gmail
    gmail_credentials_path: str = "credentials.json"
    gmail_token_path: str = "token.json"

    # Pipeline
    lookback_days: int = 90
    confidence_threshold: float = 0.5


def get_settings() -> Settings:
    """Factory function for settings (enables testing with overrides)."""
    return Settings()  # type: ignore[call-arg]
