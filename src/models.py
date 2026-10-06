from pydantic import BaseModel


class TranscriptSegment(BaseModel):
    index: int          # 0-based, ordered
    start: float        # seconds, global (offset-applied)
    end: float          # seconds, global
    text: str


class Transcript(BaseModel):
    schema_version: str = "1.0"
    source_id: str                    # slug, e.g. "pypl_q4_2025_earnings_call"
    source_file: str                  # original filename
    duration_seconds: float
    transcription_model: str          # "whisper-1" or "uploaded"
    language: str = "en"
    created_at: str                   # ISO-8601 UTC
    segments: list[TranscriptSegment]


class TextChunk(BaseModel):
    id: str            # f"{source_id}::{chunk_index}"
    source_id: str
    chunk_index: int
    start: float
    end: float
    text: str


class RetrievedChunk(BaseModel):
    text: str
    start: float
    end: float
    score: float       # cosine distance from Chroma (lower = closer)
    chunk_index: int


class SourceInfo(BaseModel):
    source_id: str
    source_file: str
    chunk_count: int
    indexed_at: float          # unix timestamp; used to find the latest upload


class IngestResult(BaseModel):
    transcript: Transcript
    chunk_count: int
    json_path: str
    srt_path: str
