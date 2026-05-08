"""Ollama Embeddings adapter — lightweight query-time embeddings.

Uses ``langchain-ollama``'s ``OllamaEmbeddings`` to embed user queries via
the same Ollama server that runs the LLM.  This keeps PyTorch / HuggingFace
entirely out of the serving process, freeing GPU VRAM for the LLM.

The Ollama bge-m3 model produces vectors identical to the HuggingFace version
used at index time (same weights, F16 GGUF), so ChromaDB similarity search
works correctly across the two paths.
"""
from __future__ import annotations

import logging

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app.core.config import Settings

logger = logging.getLogger(__name__)


class OllamaEmbeddingsProvider:
    """EmbeddingsProvider adapter backed by a local Ollama server."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self) -> Embeddings:
        logger.info(
            "Using Ollama embeddings at %s (model=%s)",
            self._settings.OLLAMA_HOST,
            self._settings.EMBEDDING_MODEL,
        )
        return OllamaEmbeddings(
            model=self._settings.EMBEDDING_MODEL,
            base_url=self._settings.OLLAMA_HOST,
        )
