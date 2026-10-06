"""Transcript → time-aware text chunks → vector index.

Long earnings-call transcripts are split into overlapping character windows so a
retrieved chunk stays readable on its own while still pointing back at the audio
via its ``[start, end]`` span. Chunks are embedded through an injected OpenAI-like
client and upserted into ChromaDB keyed by ``f"{source_id}::{chunk_index}"``.

Settings are read defensively with ``getattr`` so both ``settings=None`` (unit
tests) and partial settings objects (the pipeline layer) work unchanged.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from src.embeddings import embed_texts
from src.models import TextChunk, Transcript
from src.vector_store import upsert_chunks


def chunk_transcript(
    t: Transcript,
    target_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[TextChunk]:
    """Split ``t`` into overlapping chunks that keep their time span.

    Consecutive segment text is accumulated until the buffer reaches
    ``target_chars``; the emitted chunk spans the buffer's ``[start, end]``. The
    next buffer starts with the trailing ``overlap_chars`` characters and
    inherits the start time of the segment the overlap came from.

    A single segment longer than ``target_chars`` is hard-split into
    ``target_chars - overlap_chars`` sized pieces, each inheriting that segment's
    full ``[start, end]`` so no text is lost.
    """
    chunks: list[TextChunk] = []
    source_id = t.source_id

    def emit(text: str, start: float, end: float) -> None:
        n = len(chunks)
        chunks.append(
            TextChunk(
                id=f"{source_id}::{n}",
                source_id=source_id,
                chunk_index=n,
                start=start,
                end=end,
                text=text,
            )
        )

    buffer = ""
    buffer_start: float | None = None
    buffer_end: float | None = None
    # (char offset in buffer, start time of the segment that began there)
    buffer_spans: list[tuple[int, float]] = []

    for segment in t.segments:
        text = segment.text
        if not text:
            continue

        # A single oversized segment can't be windowed across segments; flush
        # whatever is pending, then emit fixed-size pieces over its full span.
        if len(text) > target_chars:
            if buffer:
                emit(buffer, buffer_start, buffer_end)
                buffer = ""
                buffer_spans = []
                buffer_start = None
                buffer_end = None
            step = target_chars - overlap_chars
            if step <= 0:
                step = target_chars if target_chars > 0 else len(text)
            for offset in range(0, len(text), step):
                emit(text[offset:offset + step], segment.start, segment.end)
            continue

        if not buffer:
            buffer_start = segment.start
        buffer_spans.append((len(buffer), segment.start))
        buffer += text
        buffer_end = segment.end

        if len(buffer) >= target_chars:
            emit(buffer, buffer_start, buffer_end)

            keep = buffer[-overlap_chars:] if overlap_chars > 0 else ""
            if keep:
                cut = len(buffer) - len(keep)
                # The overlap's start time is that of the segment covering its
                # first character (the last span starting at or before ``cut``).
                overlap_start = buffer_start
                for offset, span_start in buffer_spans:
                    if offset <= cut:
                        overlap_start = span_start
                    else:
                        break
                # Keep the finer segment boundaries that fall inside the overlap
                # so later windows inherit exact start times.
                kept_spans = [(0, overlap_start)]
                kept_spans.extend(
                    (offset - cut, span_start)
                    for offset, span_start in buffer_spans
                    if offset > cut
                )
                buffer = keep
                buffer_start = overlap_start
                buffer_spans = kept_spans
            else:
                buffer = ""
                buffer_start = None
                buffer_spans = []

    if buffer:
        emit(buffer, buffer_start, buffer_end)

    return chunks


def index_transcript(
    t: Transcript,
    collection,
    client,
    settings,
    indexed_at: float | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    """Chunk, embed and upsert ``t``; return the number of chunks written.

    ``settings`` may be ``None`` or a partial object: model and batch size fall
    back to the defaults. ``indexed_at`` defaults to the current time; pass an
    explicit value to group chunks from one ingest run.
    """
    target_chars = getattr(settings, "chunk_target_chars", 1200)
    overlap_chars = getattr(settings, "chunk_overlap_chars", 200)
    chunks = chunk_transcript(t, target_chars=target_chars, overlap_chars=overlap_chars)
    if not chunks:
        return 0

    vectors = embed_texts(
        [chunk.text for chunk in chunks],
        client=client,
        model=getattr(settings, "embedding_model", "text-embedding-3-small"),
        batch_size=getattr(settings, "embed_batch_size", 100),
    )
    upsert_chunks(
        collection,
        chunks,
        vectors,
        indexed_at=time.time() if indexed_at is None else indexed_at,
        source_file=t.source_file,
    )

    if progress is not None:
        progress(len(chunks), len(chunks))
    return len(chunks)
