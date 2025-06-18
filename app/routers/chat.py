import asyncio

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from llama_index.core.query_engine import BaseQueryEngine
from pydantic import BaseModel

from app.services import llm_service

router = APIRouter(
    prefix="/qa",
    tags=["qa"],
    responses={404: {"description": "Not found"}},
)


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


async def answer_streamer(response_gen):
    # Assume response_gen is an async or sync generator yielding text chunks
    for chunk in response_gen:
        yield chunk
        await asyncio.sleep(0)  # Yield control to event loop


@router.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """
    Ask a question and get an answer.
    """
    if not question.question:
        raise HTTPException(status_code=400, detail="Question is required")

    try:
        query_engine = llm_service.query_engine
        if not query_engine:
            raise HTTPException(status_code=500, detail="LLM service is not available")

        if isinstance(query_engine, BaseQueryEngine):
            streaming_response = query_engine.query(
                question.question,
            )
            if streaming_response is None:
                raise HTTPException(
                    status_code=500, detail="Failed to get a response from the model"
                )

            return StreamingResponse(
                answer_streamer(streaming_response.response_gen),  # type: ignore
                media_type="text/plain",
            )

        return Answer(answer="model works")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
