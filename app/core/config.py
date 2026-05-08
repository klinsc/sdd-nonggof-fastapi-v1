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

    # Ollama server (shared by LLM, embeddings, and OCR)
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_OCR_MODEL: str = "qwen2.5-vl"  # vision model for scripts/run_local_ocr.py

    DATASET_NAME: str = "sdd-data"
    CHROMA_DIR: str = "storage/chroma_data"
    SQLITE_PATH: str = "storage/app.sqlite3"

    # LLM provider: "cloud" = OpenAI API, "local" = Ollama
    LLM_PROFILE: Literal["cloud", "local"] = "local"
    LLM_MODEL: str = "qwen2.5:32b"  # Ollama model name or OpenAI model name

    # Embeddings: "ollama" = via Ollama server, "huggingface" = via PyTorch
    EMBEDDING_PROFILE: Literal["ollama", "huggingface"] = "ollama"
    EMBEDDING_MODEL: str = "bge-m3:latest"  # Ollama model or HuggingFace model name

    # Retrieval
    RETRIEVAL_K: int = 6
    RETRIEVAL_SCORE_THRESHOLD: float = 1.5  # Chroma L2 distance; lower = more similar
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
