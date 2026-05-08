from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.core.config import Settings


class LlamaCppLocalProvider:
    """Air-gapped LLM profile.

    The legacy LlamaIndex-based pipeline lives in `app/services/llm_service.py`.
    This adapter is a placeholder: the air-gapped roadmap intends to expose
    a local GGUF model (e.g. Typhoon 2.1 Gemma3 4B) behind the same port.
    Wire it up when the intranet deployment requires it.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self) -> BaseChatModel:  # pragma: no cover - not yet implemented
        raise NotImplementedError(
            "LlamaCppLocalProvider is not yet implemented. "
            "See app/services/llm_service.py for the legacy reference and "
            "wire a langchain ChatLlamaCpp (or equivalent) here."
        )
