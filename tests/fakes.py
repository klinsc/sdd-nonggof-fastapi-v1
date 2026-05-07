"""Test doubles for the application's ports.

These let us exercise the API and use case without touching OpenAI, GPUs,
HuggingFace downloads, or Chroma on disk.
"""
from __future__ import annotations

from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))]


class FakeVectorStoreHandle:
    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    def similarity_search(self, query: str, k: int) -> list[Document]:
        return self._docs[:k]


class FakeVectorStoreRepository:
    def __init__(self, docs: list[Document] | None = None) -> None:
        self._docs = docs or [
            Document(page_content="fake context about substations", metadata={"source": "fake"})
        ]
        self._exists = True

    def exists(self) -> bool:
        return self._exists

    def open(self) -> FakeVectorStoreHandle:
        return FakeVectorStoreHandle(self._docs)

    def build(self, documents: Iterable[Document]) -> FakeVectorStoreHandle:
        self._docs = list(documents)
        self._exists = True
        return FakeVectorStoreHandle(self._docs)


def fake_llm() -> FakeListChatModel:
    return FakeListChatModel(responses=["สวัสดี (test response)"])


class RecordingQueryLog:
    def __init__(self) -> None:
        self.records: list[tuple[str, str | None, dict | None]] = []

    def save(self, content, session_id=None, metadata=None) -> None:
        self.records.append((content, session_id, metadata))
