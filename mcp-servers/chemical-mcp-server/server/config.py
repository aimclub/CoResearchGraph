from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PACKAGE_ROOT / ".env"


class Settings(BaseSettings):
    """All environment-driven settings for the chemical MCP server."""

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Chemical service
    chem_services_host: str = Field(default="localhost")
    chem_services_port: int = Field(default=8005)
    chem_services_timeout: int = Field(default=60)

    # Retrosynthesis
    retrosynthesis_services_host: str = Field(default="localhost")
    retrosynthesis_services_port: int = Field(default=8001)
    retrosynthesis_request_timeout: int = Field(default=60)

    # S3-compatible storage. S3__* is the name every server and the main app use
    # (CoScientist/config/settings.py). The old S3_* names stay as a fallback for
    # one release, so an existing deployment keeps its credentials.
    s3_endpoint_url: str | None = Field(
        default=None, validation_alias=AliasChoices("S3__ENDPOINT_URL", "S3_ENDPOINT_URL"))
    s3_bucket_name: str | None = Field(
        default=None, validation_alias=AliasChoices("S3__BUCKET_NAME", "S3_BUCKET_NAME"))
    s3_access_key: str | None = Field(
        default=None, validation_alias=AliasChoices("S3__ACCESS_KEY", "S3_ACCESS_KEY"))
    s3_secret_key: str | None = Field(
        default=None, validation_alias=AliasChoices("S3__SECRET_KEY", "S3_SECRET_KEY"))

    chem_mcp_host: str = Field(default="0.0.0.0")
    chem_mcp_port: int = Field(default=7331)
    chem_mcp_path: str = Field(default="/mcp")


def get_settings() -> Settings:
    """Load and validate settings (cached). Call from `main()` for fail-fast startup."""
    return Settings()
