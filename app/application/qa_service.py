"""QA Service — LangGraph RAG pipeline for PEA substation standards.

Forces retrieval on every query (no LLM-decided skip), filters results by
score threshold, and generates Thai-language responses citing source documents.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, MessagesState, StateGraph

from app.application.ports import VectorStoreHandle, VectorStoreRepository

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยตอบคำถามด้านมาตรฐานการออกแบบสถานีไฟฟ้า 115kV ของการไฟฟ้าส่วนภูมิภาค (กฟภ.)\n"
    "ใช้เฉพาะข้อมูลที่ค้นพบจากเอกสารมาตรฐานด้านล่างในการตอบคำถาม\n\n"
    "กฎ:\n"
    "1. ตอบเป็นภาษาไทยเท่านั้น\n"
    "2. อ้างอิงแหล่งที่มา (ชื่อไฟล์เอกสาร) ทุกครั้งที่ตอบ\n"
    "3. ถ้าไม่มีข้อมูลที่เกี่ยวข้อง ให้ตอบว่า 'ไม่พบข้อมูลที่เกี่ยวข้องในเอกสารมาตรฐาน'\n"
    "4. ใช้คำศัพท์วิศวกรรมภาษาอังกฤษตามต้นฉบับ เช่น 115kV, Transformer, Bus Bar\n"
    "5. ตอบกระชับ ไม่เกิน 10 ประโยค\n\n"
    "เอกสารอ้างอิง:\n"
)

_NO_CONTEXT_MSG = (
    "ไม่พบข้อมูลที่เกี่ยวข้องเพียงพอในเอกสารมาตรฐาน กรุณาลองถามใหม่ด้วยคำค้นที่เฉพาะเจาะจงมากขึ้น"
)


def _coerce_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content)


class QAService:
    """Use case: orchestrates retrieval + generation as a LangGraph.

    The graph is built once at startup.  Retrieval is **forced** on every
    query — the LLM never decides whether to skip it.  Retrieved documents
    are filtered by score threshold before being injected into the prompt.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        vector_store_repo: VectorStoreRepository,
        retrieval_k: int,
        score_threshold: float = 1.5,
    ) -> None:
        self._llm = llm
        self._vector_store_repo = vector_store_repo
        self._retrieval_k = retrieval_k
        self._score_threshold = score_threshold
        self._cached_handle: VectorStoreHandle | None = None
        self._graph = self._build_graph()

    def _vector_store(self) -> VectorStoreHandle:
        if self._cached_handle is None:
            if not self._vector_store_repo.exists():
                raise RuntimeError(
                    "Vector store has not been built. "
                    "Run `python -m app.ingestion.build_index` first."
                )
            self._cached_handle = self._vector_store_repo.open()
        return self._cached_handle

    def _retrieve_and_filter(self, query: str) -> tuple[str, list[Document]]:
        """Fetch k candidates, filter by score threshold, return serialized context."""
        handle = self._vector_store()
        results = handle.similarity_search_with_score(query, k=self._retrieval_k)

        # Filter by score threshold (Chroma L2 distance: lower = better)
        filtered = [
            (doc, score)
            for doc, score in results
            if score <= self._score_threshold
        ]

        logger.info(
            "Retrieval: %d candidates, %d passed threshold (%.2f)",
            len(results),
            len(filtered),
            self._score_threshold,
        )

        if not filtered:
            return _NO_CONTEXT_MSG, []

        docs = [doc for doc, _ in filtered]
        serialized = "\n\n".join(
            f"[แหล่งอ้างอิง: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )
        return serialized, docs

    def _build_graph(self):
        llm = self._llm
        retrieve_fn = self._retrieve_and_filter

        def retrieve(state: MessagesState):
            """Always retrieve context for the latest human message."""
            # Find the latest human message
            human_msg = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    human_msg = _coerce_text(msg.content)
                    break

            context, docs = retrieve_fn(human_msg)

            # Inject context as a system message
            system = SystemMessage(_SYSTEM_PROMPT + context)

            return {"messages": [system]}

        def generate(state: MessagesState):
            """Generate a response using the LLM with retrieved context."""
            # Collect system + human messages (skip intermediate state)
            messages = [
                m for m in state["messages"]
                if isinstance(m, (SystemMessage, HumanMessage))
                or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))
            ]
            response = llm.invoke(messages)
            return {"messages": [response]}

        builder = StateGraph(MessagesState)
        builder.add_node("retrieve", retrieve)
        builder.add_node("generate", generate)
        builder.set_entry_point("retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def latest_human_text(self, input_message: str) -> str:
        return _coerce_text(input_message)

    async def astream(self, input_message: str) -> AsyncIterator[tuple[Any, dict]]:
        state = MessagesState(messages=[HumanMessage(content=input_message)])
        async for chunk in self._graph.astream(state, stream_mode="messages"):
            yield chunk
