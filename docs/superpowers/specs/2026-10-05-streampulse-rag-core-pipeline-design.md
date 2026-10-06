# StreamPulse-RAG — Core MVP Pipeline Design

- **Date:** 2026-10-05
- **Status:** Approved (design), pending spec review → implementation plan
- **Scope:** Core MVP pipeline (audio → transcript → embeddings → ChromaDB → RAG Q&A → Streamlit UI)

## 1. Context

StreamPulse-RAG turns user-uploaded FinTech earnings-call MP3s into timestamped,
searchable intelligence. This spec covers the first end-to-end MVP: local audio
upload, timestamped transcription, persistent vector indexing, and a Q&A interface
with timestamped citations.

The project already has (committed): `AGENTS.md`, `requirements.txt`, `.gitignore`,
`.env.example`, `smoke_test.py`, and a Python 3.13 virtual environment with all
dependencies installed. Repository is a git repo (branch `main`).

## 2. Goals

- Upload an earnings-call MP3 in the browser and produce a **timestamped transcript**.
- Allow the transcript to be **downloaded, and later re-uploaded**, so re-transcription
  is optional.
- **Persist** transcript chunks + embeddings in a local ChromaDB.
- Answer natural-language questions grounded in the call with **timestamped citations**.
- Keep the code **modular**, unit-testable, and runnable via `streamlit run app.py`.

## 3. Non-Goals (Explicitly Out of Scope for MVP)

- Speaker diarization / speaker labels (decided: ignored — see §11).
- Multi-user auth, remote/hosted deployment, background job queue.
- Real-time/streaming transcription.
- Cross-file comparison dashboards (only per-source Q&A + source management).
- Speech-to-text of formats other than what `pydub`/`ffmpeg` can decode.

## 4. Success Criteria

1. `streamlit run app.py` starts a working UI.
2. Uploading `data/PYPL_Q4_2025_Earnings_Call.mp3` (or a short excerpt) produces a
   transcript with sensible `start`/`end` timestamps.
3. The transcript downloads as JSON and SRT, and a downloaded JSON can be re-uploaded
   to skip transcription.
4. Indexing persists to `chroma_db/` and survives an app restart.
5. A natural-language question returns an answer whose context includes correct
   `[mm:ss]` timestamps.
6. `pytest` sanity suite passes without an API key (key-dependent tests auto-skip).

## 5. Architecture Overview

```
                 ┌──────────────────────── Streamlit app.py ────────────────────────┐
                 │  Sidebar: key status, chat model, chunk size, top-k, sources      │
                 │  Tab A: Upload MP3 → player → Transcribe → preview → Download → Index │
                 │  Tab B: Upload Transcript (JSON/SRT) → preview → Index             │
                 │  Chat: question → answer + timestamped citations                   │
                 └───────┬───────────────────────────────┬──────────────────────────┘
                         │                               │
                 pipeline.ingest_audio            pipeline.ingest_transcript
                         │                               │
      audio.split_audio │                               │
      transcribe.*      │                               │
      transcript_io.*   ▼                               ▼
                   Transcript (Pydantic) ───► indexer.chunk_transcript
                                                     │
                                              embeddings.embed_texts
                                                     │
                                            vector_store.upsert_chunks
                                                     │
                                                     ▼
                                              ChromaDB (persistent)

      Question ─► embeddings.embed_query ─► vector_store.query ─► rag.answer_question ─► Answer + citations
```

## 6. Module Layout

```
app.py                       # Streamlit UI (thin orchestrator)
src/
  __init__.py
  config.py                  # env loading, paths, model names, constants
  models.py                  # Pydantic data contracts
  audio.py                   # pydub probe + split into timestamped 5-min chunks
  transcribe.py              # whisper-1 transcription + offset merge + progress
  transcript_io.py           # JSON save/load, SRT export, upload parsing
  embeddings.py              # text-embedding-3-small (batched)
  vector_store.py            # ChromaDB persistent client operations
  indexer.py                 # transcript -> time-aware text chunks -> embeddings -> Chroma
  rag.py                     # retrieval + GPT-4o answer with timestamped citations
  pipeline.py                # end-to-end orchestration + progress callbacks
tests/
  conftest.py
  test_audio.py
  test_transcript_io.py
  test_indexer.py
  test_vector_store.py
  test_rag.py
  test_e2e_excerpt.py        # key-gated; skips without OPENAI_API_KEY
docs/superpowers/specs/      # this spec
chroma_db/                   # runtime vector store (gitignored)
transcripts/                 # saved transcript artifacts (runtime; JSON/SRT gitignored)
```

## 7. Data Contracts (`src/models.py`)

### 7.1 `TranscriptSegment`
```python
class TranscriptSegment(BaseModel):
    index: int          # 0-based, ordered
    start: float        # seconds, global (offset-applied)
    end: float          # seconds, global
    text: str
```

### 7.2 `Transcript` (the downloadable / re-uploadable artifact)
```python
class Transcript(BaseModel):
    schema_version: str = "1.0"
    source_id: str                    # slug, e.g. "pypl_q4_2025_earnings_call"
    source_file: str                  # original filename
    duration_seconds: float
    transcription_model: str          # "whisper-1" or "uploaded"
    language: str = "en"
    created_at: str                   # ISO-8601 UTC
    segments: list[TranscriptSegment]
```

Canonical JSON example:
```json
{
  "schema_version": "1.0",
  "source_id": "pypl_q4_2025_earnings_call",
  "source_file": "PYPL_Q4_2025_Earnings_Call.mp3",
  "duration_seconds": 3724.03,
  "transcription_model": "whisper-1",
  "language": "en",
  "created_at": "2026-10-05T00:00:00Z",
  "segments": [
    {"index": 0, "start": 0.0, "end": 4.2, "text": "..."}
  ]
}
```

### 7.3 `TextChunk` (unit embedded into Chroma)
```python
class TextChunk(BaseModel):
    id: str            # f"{source_id}::{chunk_index}"
    source_id: str
    chunk_index: int
    start: float
    end: float
    text: str
```

### 7.4 `RetrievedChunk`
```python
class RetrievedChunk(BaseModel):
    text: str
    start: float
    end: float
    score: float       # cosine distance from Chroma (lower = closer)
    chunk_index: int
```

### 7.5 `SourceInfo` and `IngestResult`
```python
class SourceInfo(BaseModel):
    source_id: str
    source_file: str
    chunk_count: int

class IngestResult(BaseModel):
    transcript: Transcript
    chunk_count: int
    json_path: str
    srt_path: str
```

### 7.6 Chroma metadata (scalars only)
`source_id` (str), `chunk_index` (int), `start` (float), `end` (float).
Document = chunk `text`. Collection space = cosine.

## 8. Module Interfaces & Behavior

### 8.1 `config.py`
- `load_settings() -> Settings` (dataclass) reading `.env` via `python-dotenv`.
- Fields: `openai_api_key`, `chat_model` (`gpt-4o`), `transcription_model` (`whisper-1`),
  `embedding_model` (`text-embedding-3-small`), `chroma_dir` (`chroma_db`),
  `collection_name` (`earnings_calls`), `audio_chunk_seconds` (300),
  `embed_batch_size` (100), `rag_top_k` (6), `chunk_target_chars` (1200),
  `chunk_overlap_chars` (200).
- `Settings.has_api_key` property. Never logs the key.

### 8.2 `audio.py` (pydub)
- `probe_duration(path: Path) -> float`
- `split_audio(path, chunk_seconds=300, limit_seconds=None, out_dir=None,
  sample_rate=16000, bitrate="64k") -> list[AudioChunk]`
  - `AudioChunk` (dataclass): `index`, `start`, `end`, `path` (temp exported file).
  - Re-encodes each chunk to **mono 16 kHz MP3** so a 5-min chunk ≈ 2.4 MB
    (well under OpenAI's 25 MB/request cap, even from the 190 MB Adyen source).
  - `limit_seconds` truncates to the first N seconds (powers the excerpt test).
  - Chunk `start`/`end` are **global offsets** into the original audio.
- `cleanup_chunks(chunks)` removes temp files.

### 8.3 `transcribe.py` (OpenAI, whisper-1 only)
- `transcribe_chunk(path, model, client) -> list[LocalSegment]`
  - Calls `client.audio.transcriptions.create(file=..., model="whisper-1",
    response_format="verbose_json", timestamp_granularities=["segment"])`.
  - Returns `{start, end, text}` in **chunk-local** time.
- `transcribe_audio(chunks, model, client, progress: Callable[[int,int],None] | None)
  -> Transcript`
  - Offsets each chunk's segments by the chunk's global `start`, concatenates,
    renumbers `index`, sets `duration_seconds` from the source probe.
- Retry with exponential backoff on transient API errors (2 retries).

### 8.4 `transcript_io.py`
- `save_transcript(t, out_dir="transcripts") -> Path` (canonical JSON).
- `load_transcript(path) -> Transcript`.
- `transcript_to_srt(t) -> str` (no speaker prefixes).
- `parse_transcript_file(filename, data: bytes) -> Transcript`
  - JSON canonical file → `Transcript`; SRT → best-effort parse with
    `transcription_model="uploaded"`; unknown → raise `TranscriptFormatError`.
- `slugify(name) -> str` for `source_id`.

### 8.5 `embeddings.py`
- `embed_texts(texts, model, api_key, batch_size=100) -> list[list[float]]`
- `embed_query(text, model, api_key) -> list[float]`
- Uses the OpenAI SDK directly; batches requests; raises `EmbeddingError` on failure.

### 8.6 `vector_store.py`
- `get_client(persist_dir) -> chromadb.ClientAPI` (`PersistentClient`).
- `get_or_create_collection(client, name) -> Collection` (cosine space).
- `upsert_chunks(collection, chunks, embeddings) -> None` (idempotent by `id`).
- `query(collection, query_embedding, top_k, source_id=None) -> list[RetrievedChunk]`
  (optional `where={"source_id": source_id}` filter).
- `list_sources(collection) -> list[SourceInfo]` (source_id, filename, chunk count).
- `delete_source(collection, source_id) -> None`.
- `count(collection) -> int`.

### 8.7 `indexer.py`
- `chunk_transcript(t, target_chars=1200, overlap_chars=200) -> list[TextChunk]`
  - Groups consecutive segments into ~`target_chars` windows with ~`overlap_chars`
    overlap, preserving each window's `[start, end]` span.
  - A single very long segment is hard-split while keeping its time span.
- `index_transcript(t, collection, settings, progress=None) -> int`
  - Builds chunks → embeds → upserts; returns chunk count.

### 8.8 `rag.py`
- `build_context(retrieved) -> str` formatting each chunk as
  `[mm:ss–mm:ss] text`.
- `SYSTEM_PROMPT` instructs GPT-4o to answer **only** from context and to cite
  `[mm:ss]` timestamps; to say when context is insufficient.
- `answer_question(question, collection, settings, chat_model, top_k, source_id=None)
  -> tuple[str, list[RetrievedChunk]]`
  - Embeds question → `vector_store.query` → build messages → `ChatOpenAI`
    (`langchain-openai`) `.invoke` → returns `(answer, retrieved)` for UI citations.

### 8.9 `pipeline.py`
- `ingest_audio(audio_path, collection, settings, limit_seconds=None, progress=None)
  -> IngestResult`: split → transcribe → save JSON/SRT → index.
- `ingest_transcript(transcript, collection, settings, progress=None) -> int`.
- Progress is reported as `(stage: str, current: int, total: int)` for the UI.

### 8.10 `app.py` (Streamlit)
- **Sidebar:** API-key status (env or password input), chat model select, chunk
  length (seconds), top-k; indexed-source list with delete buttons.
- **Tab "Transcribe Audio":** `st.file_uploader` (mp3/wav) → `st.audio` player →
  "Transcribe" button with `st.progress` → transcript preview (first N segments) →
  "Download JSON" / "Download SRT" → "Index to vector store".
- **Tab "Upload Transcript":** `st.file_uploader` (json/srt) → preview → Index.
- **Chat:** `st.chat_input`; renders answer with `[mm:ss]` citations and expandable
  retrieved source segments.
- State in `st.session_state` (`transcript`, `collection`, `messages`).
- All actions guarded: if no API key, show a clear message and disable network actions.

## 9. Configuration (`.env` / `.env.example`)

```
OPENAI_API_KEY=
OPENAI_MODEL_NAME=gpt-4o
TRANSCRIPTION_MODEL=whisper-1
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_DIR=chroma_db
AUDIO_CHUNK_SECONDS=300
```

## 10. Error Handling

- Missing/invalid API key → clear UI message; transcription/indexing disabled.
- Unsupported or undecodable audio → `AudioError` surfaced in the UI.
- Exported chunk exceeding 25 MB → logged + raised (should not happen at 16 kHz mono).
- Transient OpenAI errors → bounded exponential backoff, then a friendly error.
- Malformed uploaded transcript → `TranscriptFormatError` with a helpful message.
- Chroma failures → surfaced; no silent data loss.

## 11. Decisions & Known Limitations

- **No speaker labels** (product decision). `gpt-4o-transcribe-diarize` is therefore
  unused; adding diarization later is a localized change (one branch + one metadata field).
- Timestamps are **segment-level** (whisper-1 `verbose_json`). Word-level timestamps
  are available later if tighter citations are wanted.
- `whisper-1` is required for timestamps; the `gpt-4o-transcribe` family does not
  return them.
- `langchain-community` is retained in requirements but the pipeline uses the raw
  `chromadb` client (the LangChain Chroma wrapper emits sunset warnings and obscures
  metadata handling).
- `transcripts/` artifacts are runtime output; add `transcripts/` to `.gitignore`.

## 12. Testing Strategy (pytest)

- `test_audio.py`: generate a short tone MP3 with pydub, assert chunk count/offsets
  for 300 s and `limit_seconds`.
- `test_transcript_io.py`: JSON save→load round-trip; SRT export contains timestamps;
  malformed input raises.
- `test_indexer.py`: chunker respects target size/overlap and preserves time spans.
- `test_vector_store.py`: upsert + query with **deterministic fake embeddings**
  (no API); source filter; delete.
- `test_rag.py`: context formatting and prompt assembly with a **stubbed LLM**.
- `test_e2e_excerpt.py`: real first ~90–120 s of the PYPL call through
  split → transcribe → index → query; **`pytest.mark.skipif` when no API key**.
- `pytest` added to `requirements.txt` as the only new dependency.

## 13. Deliverables

- `src/` modules listed in §6, `app.py`, `tests/` suite.
- Updated `requirements.txt` (add `pytest`), `.env.example` (§9), `.gitignore`
  (add `transcripts/`).
- Updated `AGENTS.md` (module map + run instructions).
- This spec, committed to `docs/superpowers/specs/`.

## 14. Open Follow-ups (post-MVP, not in this plan)

- Speaker diarization, cross-file comparisons, cost/latency dashboards, hosted deploy,
  word-level citation highlighting.
