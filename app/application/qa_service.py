from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.application.ports import VectorStoreHandle, VectorStoreRepository

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use ten sentences maximum."
    "Respond **only in Thai language**."
    "\n\n"
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

    The graph itself is built once at startup; per-request work happens in
    `astream`. Persistence and provider construction are injected — this
    class knows nothing about Chroma, OpenAI, or SQLite directly.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        vector_store_repo: VectorStoreRepository,
        retrieval_k: int,
    ) -> None:
        self._llm = llm
        self._vector_store_repo = vector_store_repo
        self._retrieval_k = retrieval_k
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

    def _build_graph(self):
        retrieval_k = self._retrieval_k
        get_store = self._vector_store
        llm = self._llm

        @tool(response_format="content_and_artifact")
        def retrieve(query: str):
            """Retrieve information related to a query."""
            handle = get_store()
            docs = handle.similarity_search(query, k=retrieval_k)
            serialized = "\n\n".join(
                f"Source: {d.metadata}\nContent: {d.page_content}" for d in docs
            )
            return serialized, docs

        llm_with_tools = llm.bind_tools([retrieve])

        def query_or_respond(state: MessagesState):
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}

        tools_node = ToolNode([retrieve])

        def generate(state: MessagesState):
            recent_tool_messages = []
            for message in reversed(state["messages"]):
                if message.type == "tool":
                    recent_tool_messages.append(message)
                else:
                    break
            tool_messages = recent_tool_messages[::-1]
            docs_content = "\n\n".join(m.content for m in tool_messages)
            system = SystemMessage(_SYSTEM_PROMPT + docs_content)

            conversation = [
                m
                for m in state["messages"]
                if m.type in ("human", "system")
                or (m.type == "ai" and not m.tool_calls)
            ]
            response = llm.invoke([system] + conversation)
            return {"messages": [response]}

        builder = StateGraph(MessagesState)
        builder.add_node("query_or_respond", query_or_respond)
        builder.add_node("tools", tools_node)
        builder.add_node("generate", generate)
        builder.set_entry_point("query_or_respond")
        builder.add_conditional_edges(
            "query_or_respond", tools_condition, {END: END, "tools": "tools"}
        )
        builder.add_edge("tools", "generate")
        builder.add_edge("generate", END)
        return builder.compile()

    def latest_human_text(self, input_message: str) -> str:
        return _coerce_text(input_message)

    async def astream(self, input_message: str) -> AsyncIterator[tuple[Any, dict]]:
        state = MessagesState(messages=[HumanMessage(content=input_message)])
        async for chunk in self._graph.astream(state, stream_mode="messages"):
            yield chunk
