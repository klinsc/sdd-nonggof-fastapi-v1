import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import START, MessagesState, StateGraph
from pydantic import BaseModel

from app.services import langgraph_service, llm_service

router = APIRouter(
    prefix="/qa",
    tags=["qa"],
    responses={404: {"description": "Not found"}},
)


class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


graph = langgraph_service.build_graph()
if not graph:
    raise HTTPException(status_code=500, detail="Graph service is not available")


async def answer_streamer(response_gen):
    # Assume response_gen is an async or sync generator yielding text chunks
    texts = ""
    for chunk in response_gen:
        if "<|eot_id|>" in texts:
            break  # Stop if end of text marker is found

        texts += chunk
        yield chunk

        await asyncio.sleep(0)  # Yield control to event loop


@router.get("/stream")
async def chat_stream(input_message: str):
    """
    Streams responses from the LangGraph agent based on user input,
    using stream_mode="messages" for token-level streaming.
    """

    if not input_message:
        raise HTTPException(status_code=400, detail="Input message is required")

    async def event_generator():
        initial_state = MessagesState(messages=[HumanMessage(content=input_message)])

        async for chunk in graph.astream(initial_state, stream_mode="messages"):
            message_token, metadata = chunk

            if isinstance(message_token, AIMessage):
                if getattr(message_token, "type", None) == "final_response":
                    yield f"event: end_of_ai_response\ndata: {json.dumps({'metadata': metadata})}\n\n"
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

            # Optionally handle other message types here

        # Yield stream_end once after the loop
        yield "event: stream_end\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Content-Type": "text/event-stream",
            "Access-Control-Allow-Origin": "*",  # Adjust as needed for CORS
        },
    )
