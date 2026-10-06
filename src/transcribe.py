"""Whisper-1 transcription with global timestamp offset merging.

The OpenAI client is injected by the caller rather than constructed here, so the
functions stay easy to test with a fake client and no API key. Each audio chunk
is transcribed in *chunk-local* time; :func:`transcribe_chunks` shifts those
segments back onto the original recording's timeline and renumbers them into a
single continuous list.
"""

from __future__ import annotations

import time

from src.models import TranscriptSegment


class TranscriptionError(Exception):
    """Raised when the transcription client fails for a chunk."""


_RETRIES = 2
_RETRY_BASE_DELAY = 0.1  # seconds; bounded so failing tests stay fast


def transcribe_chunk(path, client, model: str) -> list[dict]:
    """Transcribe one audio chunk into chunk-local segments.

    Each item is ``{"start": float, "end": float, "text": str}`` with times in
    seconds relative to the start of ``path``. A ``None`` segment list (e.g.
    silent audio) is treated as no speech.

    ``path`` may be any ``os.PathLike``; the OpenAI SDK opens and closes the
    file handle internally.
    """
    response = client.audio.transcriptions.create(
        file=path,
        model=model,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
    segments = getattr(response, "segments", None) or []
    return [
        {"start": segment.start, "end": segment.end, "text": segment.text}
        for segment in segments
    ]


def transcribe_chunks(
    chunks,
    client,
    model: str,
    progress=None,
) -> list[TranscriptSegment]:
    """Transcribe ``chunks`` and merge them into global, timestamped segments.

    Local timestamps from each chunk are offset by that chunk's global
    ``start``, then renumbered from 0. ``progress(completed, total)`` is invoked
    after each chunk when supplied.
    """
    chunk_list = list(chunks)
    total = len(chunk_list)
    merged: list[TranscriptSegment] = []

    for position, chunk in enumerate(chunk_list):
        local_segments = _transcribe_with_retry(chunk.path, client, model)
        for segment in local_segments:
            merged.append(
                TranscriptSegment(
                    index=len(merged),
                    start=chunk.start + segment["start"],
                    end=chunk.start + segment["end"],
                    text=segment["text"],
                )
            )
        if progress is not None:
            progress(position + 1, total)

    return merged


def _transcribe_with_retry(path, client, model: str) -> list[dict]:
    """Transcribe a chunk, retrying a bounded number of times before failing."""
    last_error: Exception | None = None
    for attempt in range(_RETRIES + 1):
        try:
            return transcribe_chunk(path, client, model)
        except Exception as exc:  # network/API failures, malformed responses...
            last_error = exc
            if attempt < _RETRIES:
                time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
    raise TranscriptionError(
        f"Failed to transcribe chunk {path} after {_RETRIES + 1} attempts: {last_error}"
    ) from last_error
