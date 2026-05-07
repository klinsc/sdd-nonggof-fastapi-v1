from __future__ import annotations

import logging

import torch
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import Settings

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingsProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(self) -> Embeddings:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading embedding model %s on %s", self._settings.EMBEDDING_MODEL, device
        )
        return HuggingFaceEmbeddings(
            model_name=self._settings.EMBEDDING_MODEL,
            model_kwargs={"device": device},
        )
