import json
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from app.application.qa_service import QAService
from app.application.ports import QueryLogRepository
from app.core.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/qa",
    tags=["qa"],
    responses={404: {"description": "Not found"}},
)


class ChatRequest(BaseModel):
    input_message: str = Field(..., min_length=1)
    session_id: str | None = None


def _get_qa(request: Request) -> QAService:
    qa: QAService | None = getattr(request.app.state, "qa_service", None)
    if qa is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA service is not ready.",
        )
    return qa


def _get_query_log(request: Request) -> QueryLogRepository | None:
    return getattr(request.app.state, "query_log", None)


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest, request: Request, background: BackgroundTasks
):
    """Stream the QA agent's response as Server-Sent Events."""
    settings = get_settings()
    if len(payload.input_message) > settings.MAX_INPUT_CHARS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"input_message exceeds {settings.MAX_INPUT_CHARS} characters.",
        )

    qa = _get_qa(request)
    query_log = _get_query_log(request)
    if query_log is not None:
        background.add_task(
            query_log.save,
            payload.input_message,
            payload.session_id,
            None,
        )

    async def event_generator():
        try:
            async for message_token, metadata in qa.astream(payload.input_message):
                if await request.is_disconnected():
                    logger.info("Client disconnected; aborting stream.")
                    break

                if isinstance(message_token, AIMessage):
                    if getattr(message_token, "type", None) == "final_response":
                        yield (
                            f"event: end_of_ai_response\n"
                            f"data: {json.dumps({'metadata': metadata})}\n\n"
                        )
                    elif getattr(message_token, "content", None):
                        data = {
                            "type": "ai_chunk",
                            "content": message_token.content,
                            "metadata": metadata,
                        }
                        yield f"event: message_chunk\ndata: {json.dumps(data)}\n\n"
                elif isinstance(message_token, ToolMessage):
                    data = {
                        "type": "tool_message",
                        "content": message_token.content,
                        "name": message_token.name,
                        "metadata": metadata,
                    }
                    yield f"event: tool_message\ndata: {json.dumps(data)}\n\n"
        except Exception as exc:
            logger.exception("Error while streaming graph response: %s", exc)
            err = {"type": "error", "message": "internal_error"}
            yield f"event: error\ndata: {json.dumps(err)}\n\n"
            return

        yield "event: stream_end\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
