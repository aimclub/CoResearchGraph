"""Configuration for Sapphire and the PDF download service."""
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from CoScientist.papers_processing_refactoring.app.settings import OpenAlexSettings


class SapphireSettings(BaseSettings):
    url: str = 'http://fpin-projects.ru:12280'

    model_config = SettingsConfigDict(env_prefix='SAPPHIRE_', extra='ignore')


class PDFCrawlerSettings(BaseSettings):
    url: str = 'http://fpin-projects.ru:1228'
    username: str | None = None
    password: SecretStr | None = None
    timeout: float = Field(default=600, gt=0)

    model_config = SettingsConfigDict(env_prefix='PDF_CRAWLER_', extra='ignore')
