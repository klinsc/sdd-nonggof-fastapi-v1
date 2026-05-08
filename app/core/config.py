from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    DEBUG: bool = False

    API_KEY: str = Field(
        ...,
        description="Shared secret presented by clients in the X-API-Key header.",
    )

    CORS_ORIGINS: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="JSON list of allowed CORS origins.",
    )

    OPENAI_API_KEY: str | None = None
    HF_TOKEN: str | None = None

    # Ollama OCR settings (used by build_index and scripts/run_local_ocr.py)
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5-vl"

    DATASET_NAME: str = "sdd-data"
    CHROMA_DIR: str = "storage/chroma_data"
    SQLITE_PATH: str = "storage/app.sqlite3"

    LLM_PROFILE: Literal["cloud", "local"] = "cloud"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    RETRIEVAL_K: int = 2
    CHUNK_SIZE: int = 3000
    CHUNK_OVERLAP: int = 1000

    MAX_INPUT_CHARS: int = 4000

    @model_validator(mode="after")
    def _validate_provider_secrets(self) -> "Settings":
        if self.LLM_PROFILE == "cloud" and not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROFILE=cloud."
            )
        if len(self.API_KEY) < 16:
            raise ValueError("API_KEY must be at least 16 characters.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
