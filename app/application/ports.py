from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


@runtime_checkable
class LLMProvider(Protocol):
    """Builds a LangChain chat model. Selectable per deployment profile."""

    def build(self) -> BaseChatModel: ...


@runtime_checkable
class EmbeddingsProvider(Protocol):
    """Builds the embeddings model used for indexing and retrieval."""

    def build(self) -> Embeddings: ...


@runtime_checkable
class VectorStoreRepository(Protocol):
    """Read/write access to the persistent vector index."""

    def exists(self) -> bool: ...
    def open(self) -> "VectorStoreHandle": ...
    def build(self, documents: Iterable[Document]) -> "VectorStoreHandle": ...


@runtime_checkable
class VectorStoreHandle(Protocol):
    def similarity_search(self, query: str, k: int) -> list[Document]: ...
    def similarity_search_with_score(self, query: str, k: int) -> list[tuple[Document, float]]: ...


@runtime_checkable
class QueryLogRepository(Protocol):
    """Append-only log of user queries (off the request hot path)."""

    def save(
        self,
        content: str,
        session_id: str | None = None,
        metadata: dict | None = None,
    ) -> None: ...
