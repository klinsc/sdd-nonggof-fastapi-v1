from __future__ import annotations

import logging
import os
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.config import Settings

logger = logging.getLogger(__name__)


class ChromaVectorStoreRepository:
    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        self._settings = settings
        self._embeddings = embeddings

    def exists(self) -> bool:
        return os.path.exists(self._settings.CHROMA_DIR)

    def open(self) -> "ChromaHandle":
        store = Chroma(
            collection_name=self._settings.DATASET_NAME,
            embedding_function=self._embeddings,
            persist_directory=self._settings.CHROMA_DIR,
        )
        return ChromaHandle(store)

    def build(self, documents: Iterable[Document]) -> "ChromaHandle":
        docs = list(documents)
        if not docs:
            raise ValueError("Cannot build an empty vector store.")
        logger.info("Indexing %d documents into Chroma at %s", len(docs), self._settings.CHROMA_DIR)
        store = Chroma.from_documents(
            documents=docs,
            collection_name=self._settings.DATASET_NAME,
            embedding=self._embeddings,
            persist_directory=self._settings.CHROMA_DIR,
        )
        return ChromaHandle(store)


class ChromaHandle:
    def __init__(self, store: Chroma) -> None:
        self._store = store

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self._store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self._store.similarity_search_with_score(query, k=k)
