import json
import logging
import time

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field

from app.application.qa_service import QAService
from app.application.ports import QueryLogRepository
from app.core.config import get_settings
from app.core.rate_limit import rate_limit_qa
from app.dependencies import require_api_key
from app.infrastructure import system_stats

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/qa",
    tags=["qa"],
    dependencies=[Depends(require_api_key)],
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


def _is_timeout(exc: BaseException) -> bool:
    """True if exc (or anything in its cause chain) is a request timeout.

    Covers a wedged/slow LLM backend whose request exceeds
    ``LLM_TIMEOUT_SECONDS`` — httpx raises ``TimeoutException`` (e.g.
    ``ReadTimeout``); the stdlib ``TimeoutError`` is included for parity.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return True
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
    return False


@router.post("/stream", dependencies=[Depends(rate_limit_qa)])
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
        t0 = time.perf_counter()
        final_usage: dict | None = None
        completion_chars = 0
        try:
            async for message_token, metadata in qa.astream(payload.input_message):
                if await request.is_disconnected():
                    logger.info("Client disconnected; aborting stream.")
                    break

                if isinstance(message_token, AIMessage):
                    um = getattr(message_token, "usage_metadata", None)
                    if um:
                        # langchain returns usage_metadata as TypedDict-like.
                        # Keep the latest non-empty record (Ollama emits it on
                        # the final chunk).
                        final_usage = dict(um)
                    if getattr(message_token, "type", None) == "final_response":
                        yield (
                            f"event: end_of_ai_response\n"
                            f"data: {json.dumps({'metadata': metadata})}\n\n"
                        )
                    elif getattr(message_token, "content", None):
                        completion_chars += len(message_token.content)
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
            if _is_timeout(exc):
                logger.error(
                    "LLM request timed out after %.0fs: %s",
                    settings.LLM_TIMEOUT_SECONDS,
                    exc,
                )
                err = {
                    "type": "error",
                    "code": "llm_timeout",
                    "message": (
                        "ระบบใช้เวลาประมวลผลนานเกินกำหนด "
                        "(โมเดลอาจกำลังโหลดหรือเซิร์ฟเวอร์ไม่ตอบสนอง) "
                        "กรุณาลองใหม่อีกครั้ง"
                    ),
                }
            else:
                logger.exception("Error while streaming graph response: %s", exc)
                err = {"type": "error", "code": "internal_error", "message": "internal_error"}
            yield f"event: error\ndata: {json.dumps(err, ensure_ascii=False)}\n\n"
            # Terminate symmetrically so SSE clients close cleanly instead of
            # seeing an unexpected EOF (which can trigger auto-reconnect).
            yield "event: stream_end\ndata: {}\n\n"
            return

        # Per-query stats: tokens (when Ollama reports them), wall time,
        # completion size, and a snapshot of system resources at the end of
        # the stream. Best-effort — never block stream_end on this.
        try:
            duration_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            stats_payload = {
                "duration_ms": duration_ms,
                "input_chars": len(payload.input_message),
                "completion_chars": completion_chars,
                "tokens": {
                    "prompt": final_usage.get("input_tokens") if final_usage else None,
                    "completion": final_usage.get("output_tokens") if final_usage else None,
                    "total": final_usage.get("total_tokens") if final_usage else None,
                },
                "cpu": system_stats.get_cpu_stats(),
                "memory": system_stats.get_memory_stats(),
                "gpus": system_stats.get_gpu_stats(),
            }
            yield f"event: stats\ndata: {json.dumps(stats_payload)}\n\n"
        except Exception as exc:
            logger.warning("failed to emit stats event: %s", exc)

        yield "event: stream_end\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
