from __future__ import annotations

import pytest

from app.application.qa_service import QAService
from tests.fakes import FakeVectorStoreRepository, fake_llm


@pytest.mark.asyncio
async def test_qa_service_streams_without_real_providers():
    qa = QAService(
        llm=fake_llm(),
        vector_store_repo=FakeVectorStoreRepository(),
        retrieval_k=2,
    )
    chunks: list = []
    async for chunk in qa.astream("ทดสอบ"):
        chunks.append(chunk)
    assert chunks, "QAService should yield at least one chunk"
