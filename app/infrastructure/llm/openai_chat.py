from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.core.config import Settings


class OpenAIChatProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self) -> BaseChatModel:
        if self._settings.OPENAI_API_KEY and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self._settings.OPENAI_API_KEY
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the openai LLM profile.")
        return init_chat_model(
            self._settings.LLM_MODEL, model_provider="openai"
        )
