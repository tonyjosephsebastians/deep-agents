"""
OpenAI-backed speech-to-text stage for the voice pipeline.

This implementation performs lightweight local voice activity detection (VAD)
over the incoming PCM stream, groups speech into turns, then sends each turn to
OpenAI's transcription API as an in-memory WAV file.
"""

from __future__ import annotations

import io
import math
import os
import wave
from collections import deque
from typing import AsyncIterator

from openai import AsyncOpenAI

if __package__:
    from .events import STTChunkEvent, STTEvent, STTOutputEvent
else:
    from events import STTChunkEvent, STTEvent, STTOutputEvent


class OpenAITranscriptionSTT:
    """Turn-based STT using OpenAI's transcription API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        sample_rate: int = 16000,
        model: str | None = None,
        language: str | None = None,
        speech_threshold: int | None = None,
        silence_duration_ms: int | None = None,
        preroll_chunks: int | None = None,
        max_turn_ms: int | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for speech-to-text")

        self.client = AsyncOpenAI(api_key=self.api_key)
        self.sample_rate = sample_rate
        self.model = model or os.getenv(
            "OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"
        )
        self.language = language or os.getenv("OPENAI_STT_LANGUAGE") or None
        self.speech_threshold = speech_threshold or int(
            os.getenv("OPENAI_STT_VAD_THRESHOLD", "900")
        )
        self.silence_duration_ms = silence_duration_ms or int(
            os.getenv("OPENAI_STT_SILENCE_MS", "700")
        )
        self.preroll_chunks = preroll_chunks or int(
            os.getenv("OPENAI_STT_PREROLL_CHUNKS", "3")
        )
        self.max_turn_ms = max_turn_ms or int(
            os.getenv("OPENAI_STT_MAX_TURN_MS", "15000")
        )
        self._bytes_per_chunk = int(self.sample_rate * 2 * 0.1)
        self._chunk_ms = max(1, round(self._bytes_to_ms(self._bytes_per_chunk)))

    async def stream_events(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[STTEvent]:
        pre_speech_buffer: deque[bytes] = deque(maxlen=self.preroll_chunks)
        current_turn: list[bytes] = []
        speaking = False
        silence_ms = 0
        speech_ms = 0

        async for audio_chunk in audio_stream:
            if not audio_chunk:
                continue

            is_speech = self._is_speech(audio_chunk)

            if speaking:
                current_turn.append(audio_chunk)
                speech_ms += self._bytes_to_ms(len(audio_chunk))

                if is_speech:
                    silence_ms = 0
                else:
                    silence_ms += self._bytes_to_ms(len(audio_chunk))

                if (
                    silence_ms >= self.silence_duration_ms
                    or speech_ms >= self.max_turn_ms
                ):
                    transcript = await self._transcribe_turn(current_turn)
                    if transcript:
                        # Emit a single chunk before completion so the existing UI
                        # still starts a turn cleanly even without partial STT.
                        yield STTChunkEvent.create(transcript)
                        yield STTOutputEvent.create(transcript)

                    current_turn = []
                    speaking = False
                    silence_ms = 0
                    speech_ms = 0
                    pre_speech_buffer.clear()
            else:
                pre_speech_buffer.append(audio_chunk)
                if is_speech:
                    speaking = True
                    current_turn = list(pre_speech_buffer)
                    speech_ms = self._bytes_to_ms(
                        sum(len(chunk) for chunk in current_turn)
                    )
                    silence_ms = 0
                    pre_speech_buffer.clear()

        if speaking and current_turn:
            transcript = await self._transcribe_turn(current_turn)
            if transcript:
                yield STTChunkEvent.create(transcript)
                yield STTOutputEvent.create(transcript)

    async def _transcribe_turn(self, chunks: list[bytes]) -> str:
        audio_bytes = b"".join(chunks).strip(b"\x00")
        if not audio_bytes:
            return ""

        wav_bytes = self._pcm_to_wav(audio_bytes)
        request_kwargs: dict[str, object] = {
            "file": ("turn.wav", wav_bytes, "audio/wav"),
            "model": self.model,
        }
        if self.language:
            request_kwargs["language"] = self.language

        transcription = await self.client.audio.transcriptions.create(**request_kwargs)

        if isinstance(transcription, str):
            return transcription.strip()

        text = getattr(transcription, "text", "")
        return text.strip() if text else ""

    def _pcm_to_wav(self, audio_bytes: bytes) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio_bytes)
        return buffer.getvalue()

    def _is_speech(self, audio_chunk: bytes) -> bool:
        if len(audio_chunk) < 2:
            return False

        samples = memoryview(audio_chunk).cast("h")
        if not samples:
            return False

        mean_square = sum(sample * sample for sample in samples) / len(samples)
        rms = math.sqrt(mean_square)
        return rms >= self.speech_threshold

    def _bytes_to_ms(self, byte_count: int) -> int:
        bytes_per_second = self.sample_rate * 2
        return int((byte_count / bytes_per_second) * 1000)
