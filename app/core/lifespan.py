from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.application.qa_service import QAService
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.infrastructure.embeddings.hf_bge_m3 import HuggingFaceEmbeddingsProvider
from app.infrastructure.llm.llamacpp_local import LlamaCppLocalProvider
from app.infrastructure.llm.openai_chat import OpenAIChatProvider
from app.infrastructure.persistence.query_log_repo import SqliteQueryLogRepository
from app.infrastructure.vectorstore.chroma_repo import ChromaVectorStoreRepository

logger = logging.getLogger(__name__)


def _select_llm_provider(settings):
    profile = settings.LLM_PROFILE
    if profile == "cloud":
        return OpenAIChatProvider(settings)
    if profile == "local":
        return LlamaCppLocalProvider(settings)
    raise ValueError(f"Unknown LLM_PROFILE: {profile!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(debug=settings.DEBUG)
    logger.info("Starting application; wiring adapters…")

    embeddings = HuggingFaceEmbeddingsProvider(settings).build()
    vector_store_repo = ChromaVectorStoreRepository(settings, embeddings)
    llm_provider = _select_llm_provider(settings)
    llm = llm_provider.build()

    qa_service = QAService(
        llm=llm,
        vector_store_repo=vector_store_repo,
        retrieval_k=settings.RETRIEVAL_K,
    )
    query_log = SqliteQueryLogRepository(settings)

    app.state.qa_service = qa_service
    app.state.query_log = query_log
    app.state.ready = True
    logger.info("Application ready (LLM_PROFILE=%s).", settings.LLM_PROFILE)
    try:
        yield
    finally:
        app.state.ready = False
        logger.info("Application shutting down.")
