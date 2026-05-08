"""Ollama LLM adapter — local inference via Ollama server.

Uses ``langchain-ollama``'s ``ChatOllama`` to connect to a locally running
Ollama instance.  Supports tool calling (required by the LangGraph RAG
pipeline) and returns a standard ``BaseChatModel``.
"""
from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

from app.core.config import Settings

logger = logging.getLogger(__name__)


class OllamaChatProvider:
    """LLMProvider adapter backed by a local Ollama server."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self) -> BaseChatModel:
        logger.info(
            "Connecting to Ollama LLM at %s (model=%s)",
            self._settings.OLLAMA_HOST,
            self._settings.LLM_MODEL,
        )
        return ChatOllama(
            model=self._settings.LLM_MODEL,
            base_url=self._settings.OLLAMA_HOST,
            temperature=0,
        )
