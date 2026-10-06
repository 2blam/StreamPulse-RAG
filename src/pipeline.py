"""End-to-end ingestion orchestration: audio → transcript artifacts → vector index.

This module wires the single-call pipeline together so the UI and tests have one
entry point per flow:

* :func:`ingest_audio` splits an audio file, transcribes every chunk, assembles a
  :class:`~src.models.Transcript`, cleans up the temporary chunks, then hands the
  transcript to :func:`ingest_transcript`.
* :func:`ingest_transcript` indexes an already-built transcript and persists it as
  JSON (lossless) plus SRT (human-readable) under ``transcripts/``.

``split_audio`` and ``transcribe_chunks`` are imported at module level (not called
through ``src.audio``/``src.transcribe``) so callers and tests can monkeypatch the
expensive steps with ``monkeypatch.setattr(pipeline, "split_audio", ...)``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.audio import cleanup_chunks, split_audio
from src.indexer import index_transcript
from src.models import IngestResult, Transcript
from src.transcribe import transcribe_chunks
from src.transcript_io import save_transcript, slugify, transcript_to_srt

TRANSCRIPTS_DIR = "transcripts"


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (e.g. ``...Z``)."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ingest_transcript(
    transcript: Transcript,
    collection,
    client,
    settings,
    progress=None,
) -> int:
    """Index ``transcript`` and persist its JSON/SRT artifacts.

    Returns the number of text chunks written to the vector store. ``settings``
    may be ``None`` or a partial object (see :func:`src.indexer.index_transcript`).
    """
    n = index_transcript(transcript, collection, client, settings, progress=progress)

    save_transcript(transcript, out_dir=TRANSCRIPTS_DIR)
    srt_path = Path(TRANSCRIPTS_DIR) / f"{transcript.source_id}.srt"
    srt_path.write_text(transcript_to_srt(transcript), encoding="utf-8")

    return n


def ingest_audio(
    audio_path,
    collection,
    client,
    settings,
    limit_seconds=None,
    progress=None,
) -> IngestResult:
    """Run the full audio ingestion flow for one earnings call.

    Splits ``audio_path`` into chunks, transcribes them into a single global
    timeline, saves the transcript, and indexes it. Temporary chunk files are
    removed as soon as transcription finishes, whether it succeeds or raises.
    ``limit_seconds`` transcribes only the first N seconds (useful for quick
    excerpt runs).
    """
    chunks = split_audio(
        audio_path,
        chunk_seconds=settings.audio_chunk_seconds,
        limit_seconds=limit_seconds,
    )
    try:
        segments = transcribe_chunks(
            chunks, client, settings.transcription_model, progress
        )
    finally:
        # Remove temporary chunk files even if transcription fails, so a
        # TranscriptionError does not orphan split_audio's temp directory.
        cleanup_chunks(chunks)

    duration = (
        chunks[-1].end
        if chunks
        else (segments[-1].end if segments else 0.0)
    )
    transcript = Transcript(
        source_id=slugify(Path(audio_path).name),
        source_file=Path(audio_path).name,
        duration_seconds=duration,
        transcription_model=settings.transcription_model,
        created_at=_utcnow_iso(),
        segments=segments,
    )

    chunk_count = ingest_transcript(transcript, collection, client, settings)

    return IngestResult(
        transcript=transcript,
        chunk_count=chunk_count,
        json_path=str(Path(TRANSCRIPTS_DIR) / f"{transcript.source_id}.json"),
        srt_path=str(Path(TRANSCRIPTS_DIR) / f"{transcript.source_id}.srt"),
    )
