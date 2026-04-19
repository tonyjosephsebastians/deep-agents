"""
OpenAI-backed text-to-speech stage for the voice pipeline.

This implementation synthesizes PCM audio via OpenAI's Speech API and streams
raw 24kHz PCM bytes back through the existing event protocol.
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

from openai import AsyncOpenAI

if __package__:
    from .events import TTSChunkEvent
else:
    from events import TTSChunkEvent


class OpenAITTS:
    """Queue-based TTS producer that matches the existing pipeline contract."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        voice: str | None = None,
        instructions: str | None = None,
        chunk_size: int | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for text-to-speech")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model or os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
        self.voice = voice or os.getenv("OPENAI_TTS_VOICE", "alloy")
        self.instructions = instructions or os.getenv(
            "OPENAI_TTS_INSTRUCTIONS", "Speak clearly, warmly, and conversationally."
        )
        self.chunk_size = chunk_size or int(os.getenv("OPENAI_TTS_CHUNK_SIZE", "4096"))
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._closed = False

    async def send_text(self, text: str | None) -> None:
        if self._closed or not text or not text.strip():
            return
        await self._queue.put(text)

    async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
        while True:
            text = await self._queue.get()
            if text is None:
                break

            request_kwargs: dict[str, object] = {
                "input": text,
                "model": self.model,
                "voice": self.voice,
                "response_format": "pcm",
                "stream_format": "audio",
            }

            if not self.model.startswith("tts-1") and self.instructions:
                request_kwargs["instructions"] = self.instructions

            async with self.client.audio.speech.with_streaming_response.create(
                **request_kwargs
            ) as response:
                async for audio_chunk in response.iter_bytes(self.chunk_size):
                    if audio_chunk:
                        yield TTSChunkEvent.create(audio_chunk)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None)
