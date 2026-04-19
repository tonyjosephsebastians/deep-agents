import os
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

if __package__:
    from .events import (
        AgentChunkEvent,
        AgentEndEvent,
        ToolCallEvent,
        ToolResultEvent,
        VoiceAgentEvent,
        event_to_dict,
    )
    from .openai_stt import OpenAITranscriptionSTT
    from .openai_tts import OpenAITTS
    from .utils import merge_async_iters
else:
    from events import (
        AgentChunkEvent,
        AgentEndEvent,
        ToolCallEvent,
        ToolResultEvent,
        VoiceAgentEvent,
        event_to_dict,
    )
    from openai_stt import OpenAITranscriptionSTT
    from openai_tts import OpenAITTS
    from utils import merge_async_iters


CURRENT_FILE = Path(__file__).resolve()
ENV_FILES = (
    CURRENT_FILE.parents[3] / ".env",
    CURRENT_FILE.parents[5] / ".env",
)

for env_file in ENV_FILES:
    if env_file.exists():
        load_dotenv(env_file, override=False)

if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

# Static files are served from the shared web build output.
STATIC_DIR = Path(__file__).parent.parent.parent / "web" / "dist"

if not STATIC_DIR.exists():
    raise RuntimeError(
        f"Web build not found at {STATIC_DIR}. "
        "Run 'make build-web' or 'make dev-py' from the project root."
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _is_expected_websocket_send_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "Unexpected ASGI message 'websocket.send'" in message
        or 'Cannot call "send" once a close message has been sent.' in message
    )


def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    return f"Added {quantity} x {item} to the order."


def confirm_order(order_summary: str) -> str:
    """Confirm the final order with the customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."


system_prompt = """
You are a helpful sandwich shop assistant. Your goal is to take the user's order.
Be concise and friendly.

Available toppings: lettuce, tomato, onion, pickles, mayo, mustard.
Available meats: turkey, ham, roast beef.
Available cheeses: swiss, cheddar, provolone.
Ask a short follow-up question only when the order is ambiguous or incomplete.
"""

agent = create_agent(
    model=ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")),
    tools=[add_to_order, confirm_order],
    system_prompt=system_prompt,
    checkpointer=InMemorySaver(),
)


async def _stt_stream(
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Audio (Bytes) -> Voice Events (VoiceAgentEvent)

    This stage groups the incoming PCM stream into speech turns using lightweight
    local VAD and submits each turn to OpenAI's transcription API.

    Args:
        audio_stream: Async iterator of PCM audio bytes (16-bit, mono, 16kHz)

    Yields:
        STT events (stt_chunk for partials, stt_output for final transcripts)
    """
    stt = OpenAITranscriptionSTT(sample_rate=16000)

    async for event in stt.stream_events(audio_stream):
        yield event


async def _agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events -> Voice Events (with Agent Responses)

    This function takes a stream of upstream voice agent events and processes
    them. When an stt_output event arrives, it passes the transcript to the
    LangChain agent. The agent streams back its response tokens as agent_chunk
    events. Tool calls and results are also emitted as separate events. All
    other upstream events are passed through unchanged.

    The passthrough pattern ensures downstream stages (like TTS) can observe
    all events in the pipeline, not just the ones this stage produces. This
    enables features like displaying partial transcripts while the agent is
    thinking.

    Args:
        event_stream: An async iterator of upstream voice agent events

    Yields:
        All upstream events plus agent_chunk, tool_call, and tool_result events
    """
    thread_id = str(uuid4())

    async for event in event_stream:
        yield event

        if event.type == "stt_output":
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )

            async for message, _metadata in stream:
                if isinstance(message, AIMessage):
                    if message.text:
                        yield AgentChunkEvent.create(message.text)

                    if hasattr(message, "tool_calls") and message.tool_calls:
                        for tool_call in message.tool_calls:
                            yield ToolCallEvent.create(
                                id=tool_call.get("id", str(uuid4())),
                                name=tool_call.get("name", "unknown"),
                                args=tool_call.get("args", {}),
                            )

                if isinstance(message, ToolMessage):
                    yield ToolResultEvent.create(
                        tool_call_id=getattr(message, "tool_call_id", ""),
                        name=getattr(message, "name", "unknown"),
                        result=str(message.content) if message.content else "",
                    )

            yield AgentEndEvent.create()


async def _tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events -> Voice Events (with Audio)

    This function takes a stream of upstream voice agent events and processes
    them. When agent_chunk events arrive, it sends the text to OpenAI for TTS
    synthesis. Audio is streamed back as tts_chunk events as it's generated.
    All upstream events are passed through unchanged.

    It uses merge_async_iters to combine two concurrent streams:
    - process_upstream(): Iterates through incoming events, yields them for
      passthrough, and sends buffered agent text to OpenAI for synthesis.
    - tts.receive_events(): Yields audio chunks from OpenAI as they are
      synthesized.

    Args:
        event_stream: An async iterator of upstream voice agent events

    Yields:
        All upstream events plus tts_chunk events for synthesized audio
    """
    tts = OpenAITTS()

    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        buffer: list[str] = []
        try:
            async for event in event_stream:
                yield event

                if event.type == "agent_chunk":
                    buffer.append(event.text)

                if event.type == "agent_end":
                    await tts.send_text("".join(buffer))
                    buffer = []
        finally:
            await tts.close()

    try:
        async for event in merge_async_iters(process_upstream(), tts.receive_events()):
            yield event
    finally:
        await tts.close()


pipeline = (
    RunnableGenerator(_stt_stream)
    | RunnableGenerator(_agent_stream)
    | RunnableGenerator(_tts_stream)
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream() -> AsyncIterator[bytes]:
        """Async generator that yields audio bytes from the websocket."""
        while True:
            try:
                data = await websocket.receive_bytes()
            except WebSocketDisconnect:
                return
            yield data

    output_stream = pipeline.atransform(websocket_audio_stream())

    try:
        async for event in output_stream:
            try:
                await websocket.send_json(event_to_dict(event))
            except WebSocketDisconnect:
                break
            except RuntimeError as exc:
                if _is_expected_websocket_send_error(exc):
                    break
                raise
    except WebSocketDisconnect:
        pass
    finally:
        aclose = getattr(output_stream, "aclose", None)
        if aclose is not None:
            await aclose()


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("main:app", app_dir=str(Path(__file__).parent), port=8000, reload=True)
