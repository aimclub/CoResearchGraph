from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, SecretStr, model_validator

from CoScientist.papers_processing_refactoring.definitions import CONFIG_PATH


class LLMSettings(BaseSettings):
    
    llm_base_url: str
    llm_name: str
    llm_api_key: SecretStr
    
    model_config = SettingsConfigDict(
        env_prefix="ETL_",
        extra="ignore",
    )


class OpenAlexSettings(BaseSettings):

    email: str | None = None
    api_key: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_prefix="SERVICES__OPENALEX_",
        extra="ignore",
    )


class EmbeddingSettings(BaseSettings):

    type: str = Field(default="api")
    api_url: str | None = None
    model_name: str | None = None
    batch_size: int = 16

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDINGS_",
        extra="ignore",
    )


class RerankerSettings(BaseSettings):

    type: str = "api"
    api_url: str | None = None
    model_name: str | None = None
    batch_size: int = 16

    model_config = SettingsConfigDict(
        env_prefix="RERANKING_",
        extra="ignore",
    )


class ChromaSettings(BaseSettings):

    host: str = "localhost"
    port: int = 8000
    collection: str = "default_collection"

    model_config = SettingsConfigDict(
        env_prefix="CHROMADB_",
        extra="ignore",
    )


class VectorDBSettings(BaseSettings):

    backend: str = Field(default="chromadb", alias="VECTOR_DB")

    chroma: ChromaSettings = Field(default_factory=ChromaSettings)

    model_config = SettingsConfigDict(
        extra="ignore",
    )
    
    
class S3BucketSettings(BaseSettings):

    bucket: str
    
    model_config = SettingsConfigDict(extra="ignore")


class S3Settings(BaseSettings):

    endpoint: str
    access_key: str
    secret_key: SecretStr
    etl_bucket: str
    public_bucket: str

    model_config = SettingsConfigDict(
        env_prefix="S3_",
        extra="ignore",
    )


class DatabaseSettings(BaseSettings):

    type: str = Field(default="sqlite", alias="DATABASE_TYPE")
    sqlite_path: str = Field(default="./data/db.sqlite", alias="SQLITE_PATH")
    postgresql_dsn: str | None = Field(default=None, alias="POSTGRESQL_DSN")

    @model_validator(mode="after")
    def validate_postgresql_dsn(self) -> "DatabaseSettings":
        if self.type == "postgres" and not self.postgresql_dsn:
            raise ValueError("POSTGRESQL_DSN is required when DATABASE_TYPE=postgres")
        return self

    model_config = SettingsConfigDict(
        extra="ignore",
    )


class FilesSettings(BaseSettings):
    directory: Path
    
    model_config = SettingsConfigDict(
        env_prefix="FILES_",
        extra="ignore",
    )


class AppSettings(BaseSettings):
    
    embeddings: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    vectordb: VectorDBSettings = Field(default_factory=VectorDBSettings)
    s3: S3Settings = Field(default_factory=S3Settings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    openalex: OpenAlexSettings = Field(default_factory=OpenAlexSettings)
    files: FilesSettings = Field(default_factory=FilesSettings)
    
    model_config = SettingsConfigDict(
        env_file=CONFIG_PATH,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )
