from functools import lru_cache

from pydantic import Field
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
    TYHOON_API_KEY: str | None = None
    HF_TOKEN: str | None = None

    DATASET_NAME: str = "sdd-data"
    CHROMA_DIR: str = "storage/chroma_data"
    SQLITE_PATH: str = "storage/app.sqlite3"

    LLM_MODEL: str = "gpt-4o-mini"
    LLM_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"

    RETRIEVAL_K: int = 2
    CHUNK_SIZE: int = 3000
    CHUNK_OVERLAP: int = 1000

    MAX_INPUT_CHARS: int = 4000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
