from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services import llm_service

router = APIRouter(
    prefix="/qa",
    tags=["qa"],
    responses={404: {"description": "Not found"}},
)


class Question(BaseModel):
    question: str
    context: str


class Answer(BaseModel):
    answer: str


@router.post("/ask", response_model=Answer)
async def ask_question(question: Question):
    """
    Ask a question with context and get an answer.
    """
    if not question.question or not question.context:
        raise HTTPException(status_code=400, detail="Question and context are required")

    try:
        # TODO
        # answer = llm_service.answer_question(question.question, question.context)
        # return Answer(answer=answer)

        # Mock response for demonstration purposes
        answer = "This is a mock answer to your question."
        return Answer(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
