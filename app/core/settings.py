"""Operator configuration, constructed by the application lifecycle."""
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISE_", env_file=".env", extra="ignore")
    database_url: str = ""
    llm_base: str = "http://127.0.0.1:11434"
    answer_model: str = "qwen3:1.7b"
    embed_model: str = "bge-m3"
    admin_token: str = ""
    files_dir: Path = Path("data/private")
    source_dir: Path = Path("funds-data")
    fiscal_year: int = 2569
    llm_timeout: float = Field(30, gt=0, le=40)
    semantic_query_timeout: float = Field(2, gt=0, le=10)
    llm_concurrency: int = Field(1, ge=1, le=4)
    job_lease_seconds: int = Field(120, ge=15)
    job_max_attempts: int = Field(3, ge=1, le=10)
    max_pdf_bytes: int = Field(15 * 1024 * 1024, ge=1024)
    max_pdf_pages: int = Field(100, ge=1)
    ingestion_max_documents: int = Field(100, ge=1)
    ingestion_max_total_bytes: int = Field(512 * 1024 * 1024, ge=1)
    answer_enabled: bool = True
