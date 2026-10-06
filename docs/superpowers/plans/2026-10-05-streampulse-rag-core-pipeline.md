# StreamPulse-RAG Core Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local Streamlit app that transcribes uploaded FinTech earnings-call MP3s into timestamped transcripts, persists embeddings in ChromaDB, and answers questions with GPT-4o citing `[mm:ss]` timestamps.

**Architecture:** Layered `src/` modules behind Pydantic data contracts. `pydub` splits audio into 5-minute mono-16 kHz chunks; OpenAI `whisper-1` transcribes each; the transcript is saved as JSON/SRT, chunked into time-aware windows, embedded with `text-embedding-3-small`, and upserted into a persistent ChromaDB collection namespaced by `source_id`. A thin Streamlit UI orchestrates ingestion and Q&A.

**Tech Stack:** Python 3.13, Streamlit, OpenAI SDK (`whisper-1`, `gpt-4o`, `text-embedding-3-small`), ChromaDB (persistent), pydub + ffmpeg, LangChain OpenAI (`ChatOpenAI`), Pydantic, pytest.

**Spec:** `docs/superpowers/specs/2026-10-05-streampulse-rag-core-pipeline-design.md`

## Global Constraints

- Python **3.13** (venv at `venv/`); run all commands as `./venv/bin/python -m pytest ...`.
- **No speaker labels anywhere** — no `speaker` field, no diarization model.
- Transcription model is **`whisper-1`** with `response_format="verbose_json"` and `timestamp_granularities=["segment"]`.
- Embedding model is **`text-embedding-3-small`**; Chroma collection space is **cosine**.
- Audio chunks default to **300 s**, exported **mono, 16 kHz, mp3, 64 kbps** (keeps each chunk well under the 25 MB/request cap).
- `source_id = slug(filename)`; duplicate filenames overwrite that source (documented, accepted).
- Chroma metadata is scalars only: `source_id`, `source_file`, `chunk_index`, `start`, `end`, `indexed_at`.
- All timestamps are **seconds, float, global** (chunk offset applied).
- No secrets in code or logs; `OPENAI_API_KEY` comes from `.env` only.
- Add **pytest** as the only new dependency.

## Review Focus

These are the input classes most likely to bite a user, implied by the spec but not pinned by any single requirement. Each gets a test in the owning task:

1. **Silent / near-silent or sub-second audio** — must not crash; should yield an empty/valid transcript (§ Task 2/7).
2. **A transcript segment spanning a chunk boundary** — offsets must remain globally correct (§ Task 7).
3. **A single segment longer than the chunk target** — hard-split must preserve its time span and no text is lost (§ Task 6).
4. **Duplicate filename re-index** — must replace, never duplicate, that source's chunks (§ Task 5).
5. **Malformed / unknown uploaded transcript** — must raise a clear `TranscriptFormatError`, not a raw traceback (§ Task 3).

---

### Task 1: Foundation — settings, data contracts, dev deps

**Files:**
- Create: `src/__init__.py`, `src/config.py`, `src/models.py`, `tests/conftest.py`, `tests/test_config.py`, `tests/test_models.py`
- Modify: `requirements.txt`, `.gitignore`, `.env.example`

**Interfaces:**
- Consumes: nothing.
- Produces: `config.Settings`, `config.load_settings(env_file=".env") -> Settings`, `config.get_openai_client(api_key) -> openai.OpenAI`; `models.TranscriptSegment`, `models.Transcript`, `models.TextChunk`, `models.RetrievedChunk`, `models.SourceInfo`, `models.IngestResult`.

- [ ] **Step 1: Add pytest and update config files**

Append to `requirements.txt`:
```
# --- dev / test ---
pytest
audioop-lts; python_version >= "3.13"
```
Add `transcripts/` to `.gitignore`. Set `.env.example` to:
```
OPENAI_API_KEY=
OPENAI_MODEL_NAME=gpt-4o
TRANSCRIPTION_MODEL=whisper-1
EMBEDDING_MODEL=text-embedding-3-small
CHROMA_DIR=chroma_db
AUDIO_CHUNK_SECONDS=300
```
Then run: `./venv/bin/python -m pip install pytest` → Expect: `Successfully installed pytest-...`

- [ ] **Step 2: Write the failing tests**

`tests/test_config.py`:
```python
from src.config import load_settings

def test_defaults_and_has_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = load_settings(env_file=None)
    assert s.openai_api_key == "sk-test"
    assert s.chat_model == "gpt-4o"
    assert s.transcription_model == "whisper-1"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.audio_chunk_seconds == 300
    assert s.has_api_key is True

def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert load_settings(env_file=None).has_api_key is False

def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("AUDIO_CHUNK_SECONDS", "120")
    assert load_settings(env_file=None).audio_chunk_seconds == 120
```

`tests/test_models.py`:
```python
from src.models import Transcript, TranscriptSegment

def test_transcript_json_roundtrip():
    t = Transcript(source_id="x", source_file="x.mp3", duration_seconds=10.0,
                   transcription_model="whisper-1", created_at="2026-01-01T00:00:00Z",
                   segments=[TranscriptSegment(index=0, start=0.0, end=1.0, text="hi")])
    again = Transcript.model_validate_json(t.model_dump_json())
    assert again.segments[0].text == "hi"
    assert again.schema_version == "1.0"
```

`tests/conftest.py`:
```python
import pytest
from src.models import Transcript, TranscriptSegment

@pytest.fixture
def make_transcript():
    def _make(source_id="pypl", source_file="PYPL.mp3", segments=None):
        segs = segments or [TranscriptSegment(index=0, start=0.0, end=1.5, text="hello revenue")]
        return Transcript(source_id=source_id, source_file=source_file,
                          duration_seconds=segs[-1].end, transcription_model="whisper-1",
                          created_at="2026-01-01T00:00:00Z", segments=segs)
    return _make
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_config.py tests/test_models.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src'` / `src.config`.

- [ ] **Step 4: Implement**

`src/__init__.py`: empty.
`src/config.py`:
```python
from dataclasses import dataclass
import os
from dotenv import load_dotenv
from openai import OpenAI

@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    chat_model: str = "gpt-4o"
    transcription_model: str = "whisper-1"
    embedding_model: str = "text-embedding-3-small"
    chroma_dir: str = "chroma_db"
    collection_name: str = "earnings_calls"
    audio_chunk_seconds: int = 300
    embed_batch_size: int = 100
    rag_top_k: int = 6
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 200

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key.strip())

def load_settings(env_file: str | None = ".env") -> Settings:
    if env_file:
        load_dotenv(env_file)  # does not override existing env vars
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chat_model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
        transcription_model=os.getenv("TRANSCRIPTION_MODEL", "whisper-1"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chroma_dir=os.getenv("CHROMA_DIR", "chroma_db"),
        audio_chunk_seconds=int(os.getenv("AUDIO_CHUNK_SECONDS", "300")),
    )

def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)
```
`src/models.py`: Pydantic v2 `BaseModel` classes exactly as spec §7 — `TranscriptSegment(index:int, start:float, end:float, text:str)`, `Transcript(schema_version="1.0", source_id, source_file, duration_seconds, transcription_model, language="en", created_at, segments:list[TranscriptSegment])`, `TextChunk(id, source_id, chunk_index, start, end, text)`, `RetrievedChunk(text, start, end, score, chunk_index)`, `SourceInfo(source_id, source_file, chunk_count, indexed_at)`, `IngestResult(transcript, chunk_count, json_path, srt_path)`.

- [ ] **Step 5: Run tests and verify they pass**

Run: `./venv/bin/python -m pytest tests/test_config.py tests/test_models.py -q`
Expected: `3 passed` (config) + `1 passed` (models).

- [ ] **Step 6: Commit**

```bash
git add src/__init__.py src/config.py src/models.py tests/ requirements.txt .gitignore .env.example
git commit -m "feat: add settings, data contracts, and pytest dev dependency"
```

---

### Task 2: `src/audio.py` — pydub chunking with timestamp markers

**Files:**
- Create: `src/audio.py`, `tests/test_audio.py`

**Interfaces:**
- Consumes: `src/models.py` not needed here.
- Produces: `audio.AudioChunk` (dataclass: `index:int, start:float, end:float, path:Path`), `audio.AudioError`, `audio.probe_duration(path) -> float`, `audio.split_audio(path, chunk_seconds=300, limit_seconds=None, out_dir=None, sample_rate=16000, bitrate="64k") -> list[AudioChunk]`, `audio.cleanup_chunks(chunks) -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/test_audio.py`:
```python
from pathlib import Path
import pytest
from pydub import AudioSegment
from src.audio import split_audio, probe_duration, cleanup_chunks, AudioError

@pytest.fixture
def tone_mp3(tmp_path):
    p = tmp_path / "tone.mp3"
    # 12 s of a quiet tone (silence is fine for chunking tests)
    AudioSegment.silent(duration=12_000).export(p, format="mp3")
    return p

def test_probe_duration(tone_mp3):
    assert abs(probe_duration(tone_mp3) - 12.0) < 0.5

def test_split_offsets_and_files(tone_mp3, tmp_path):
    chunks = split_audio(tone_mp3, chunk_seconds=5, out_dir=tmp_path / "out")
    assert [round(c.start, 2) for c in chunks] == [0.0, 5.0, 10.0]
    assert [round(c.end, 2) for c in chunks] == [5.0, 10.0, 12.0]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert all(Path(c.path).exists() for c in chunks)
    cleanup_chunks(chunks)
    assert not Path(chunks[0].path).exists()

def test_limit_seconds(tone_mp3, tmp_path):
    chunks = split_audio(tone_mp3, chunk_seconds=5, limit_seconds=7, out_dir=tmp_path / "out")
    assert len(chunks) == 2
    assert round(chunks[-1].end, 2) == 7.0

def test_missing_file_raises(tmp_path):
    with pytest.raises(AudioError):
        split_audio(tmp_path / "nope.mp3", out_dir=tmp_path)

def test_short_audio_single_chunk(tmp_path):
    p = tmp_path / "tiny.mp3"
    AudioSegment.silent(duration=800).export(p, format="mp3")  # 0.8 s
    chunks = split_audio(p, chunk_seconds=300, out_dir=tmp_path / "out")
    assert len(chunks) == 1
    assert round(chunks[0].end, 1) <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_audio.py -q`
Expected: FAIL — `ImportError` / `cannot import name 'split_audio'`.

- [ ] **Step 3: Implement `src/audio.py`**

Approach: load with `AudioSegment.from_file`, `duration_seconds = len(audio)/1000.0`; apply `limit_seconds` via slicing; iterate `range(0, total_ms, chunk_ms)` producing `AudioChunk(index, start_ms/1000, end_ms/1000, path)`; export each slice as `format="mp3", bitrate="64k", parameters=["-ac","1","-ar","16000"]` into `out_dir` (default `tempfile.mkdtemp()`). Wrap decode/export failures in `AudioError`. `probe_duration` uses the same loader; `cleanup_chunks` unlinks files if present.

- [ ] **Step 4: Run tests and verify they pass**

Run: `./venv/bin/python -m pytest tests/test_audio.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/audio.py tests/test_audio.py
git commit -m "feat: add pydub audio chunking with timestamp offsets"
```

---

### Task 3: `src/transcript_io.py` — JSON/SRT save, load, and upload parsing

**Files:**
- Create: `src/transcript_io.py`, `tests/test_transcript_io.py`

**Interfaces:**
- Consumes: `models.Transcript`, `models.TranscriptSegment`.
- Produces: `TranscriptFormatError`; `slugify(name) -> str`; `save_transcript(t, out_dir="transcripts") -> Path`; `load_transcript(path) -> Transcript`; `transcript_to_srt(t) -> str`; `parse_transcript_file(filename, data: bytes) -> Transcript`.

- [ ] **Step 1: Write the failing tests**

`tests/test_transcript_io.py`:
```python
import pytest
from src.transcript_io import (slugify, save_transcript, load_transcript,
                               transcript_to_srt, parse_transcript_file, TranscriptFormatError)

def test_slugify():
    assert slugify("PYPL_Q4_2025_Earnings_Call.mp3") == "pypl_q4_2025_earnings_call"
    assert slugify("$XYZ Block Inc Q4 2025 [id].mp3") == "xyz_block_inc_q4_2025_id"

def test_save_load_roundtrip(make_transcript, tmp_path):
    t = make_transcript()
    p = save_transcript(t, out_dir=tmp_path)
    assert p.exists() and p.suffix == ".json"
    assert load_transcript(p).model_dump() == t.model_dump()

def test_srt_has_timestamps(make_transcript):
    srt = transcript_to_srt(make_transcript())
    assert "00:00:00,000 --> 00:00:01,500" in srt
    assert "hello revenue" in srt
    assert "Speaker" not in srt

def test_parse_uploaded_json(make_transcript):
    data = make_transcript().model_dump_json().encode()
    assert parse_transcript_file("t.json", data).source_id == "pypl"

def test_malformed_json_raises():
    with pytest.raises(TranscriptFormatError):
        parse_transcript_file("t.json", b"{not json")

def test_unknown_extension_raises():
    with pytest.raises(TranscriptFormatError):
        parse_transcript_file("t.txt", b"hello")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_transcript_io.py -q`
Expected: FAIL — import error.

- [ ] **Step 3: Implement `src/transcript_io.py`**

Notes: `slugify` lowercases, replaces any non-alphanumeric run with `_`, strips `_`. `save_transcript` writes `<out_dir>/<source_id>.json` (create dirs). `transcript_to_srt` formats `HH:MM:SS,mmm --> HH:MM:SS,mmm` then the text and a blank line, numbered from 1. `parse_transcript_file`: `.json` → `Transcript.model_validate_json`; `.srt` → best-effort parse setting `transcription_model="uploaded"` and `source_id=slugify(filename)`; any parse failure or other extension → `TranscriptFormatError`.

- [ ] **Step 4: Run tests and verify they pass**

Run: `./venv/bin/python -m pytest tests/test_transcript_io.py -q`
Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/transcript_io.py tests/test_transcript_io.py
git commit -m "feat: add transcript JSON/SRT IO and upload parsing"
```

---

### Task 4: `src/embeddings.py` — batched OpenAI embeddings

**Files:**
- Create: `src/embeddings.py`, `tests/test_embeddings.py`

**Interfaces:**
- Consumes: an injected OpenAI-like client (`client.embeddings.create(model, input)`).
- Produces: `EmbeddingError`; `embed_texts(texts, client, model, batch_size=100) -> list[list[float]]`; `embed_query(text, client, model) -> list[float]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_embeddings.py`:
```python
from types import SimpleNamespace
from src.embeddings import embed_texts, embed_query

class FakeEmbeddings:
    def __init__(self): self.calls = []
    def create(self, model, input):
        batch = list(input); self.calls.append(batch)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(i)]) for i in range(len(batch))])

class FakeClient:
    def __init__(self): self.embeddings = FakeEmbeddings()

def test_batching():
    c = FakeClient()
    out = embed_texts([f"t{i}" for i in range(250)], client=c, model="m", batch_size=100)
    assert len(out) == 250
    assert [len(b) for b in c.embeddings.calls] == [100, 100, 50]

def test_single_query():
    c = FakeClient()
    assert embed_query("hi", client=c, model="m") == [0.0]

def test_empty_input_returns_empty():
    c = FakeClient()
    assert embed_texts([], client=c, model="m") == []
    assert c.embeddings.calls == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_embeddings.py -q`
Expected: FAIL — import error.

- [ ] **Step 3: Implement `src/embeddings.py`**

Chunk `texts` into `batch_size` slices, call `client.embeddings.create(model=model, input=batch)`, extend results with `item.embedding`. Wrap exceptions in `EmbeddingError`. `embed_query` = `embed_texts([text], ...)[0]`.

- [ ] **Step 4: Run tests and verify they pass**

Run: `./venv/bin/python -m pytest tests/test_embeddings.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/embeddings.py tests/test_embeddings.py
git commit -m "feat: add batched OpenAI embeddings wrapper"
```

---

### Task 5: `src/vector_store.py` — persistent ChromaDB operations

**Files:**
- Create: `src/vector_store.py`, `tests/test_vector_store.py`

**Interfaces:**
- Consumes: `models.TextChunk`, `models.RetrievedChunk`, `models.SourceInfo`.
- Produces: `VectorStoreError`; `get_client(persist_dir) -> ClientAPI`; `get_or_create_collection(client, name) -> Collection`; `upsert_chunks(collection, chunks, embeddings, indexed_at=0.0, source_file="") -> None`; `query(collection, query_embedding, top_k, source_id=None) -> list[RetrievedChunk]`; `list_sources(collection) -> list[SourceInfo]`; `latest_source_id(collection) -> str | None`; `delete_source(collection, source_id) -> None`; `count(collection) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_vector_store.py`:
```python
import pytest
from src.models import TextChunk
from src.vector_store import (get_client, get_or_create_collection, upsert_chunks,
                              query, list_sources, latest_source_id, delete_source, count)

@pytest.fixture
def collection(tmp_path):
    return get_or_create_collection(get_client(str(tmp_path / "chroma")), "test")

def _chunk(cid, sid, i, text, start=0.0, end=5.0):
    return TextChunk(id=cid, source_id=sid, chunk_index=i, start=start, end=end, text=text)

def test_upsert_query_roundtrip(collection):
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "revenue growth")], [[1.0, 0.0, 0.0]], indexed_at=100.0)
    res = query(collection, [1.0, 0.0, 0.0], top_k=1)
    assert res[0].text == "revenue growth" and res[0].start == 0.0

def test_source_filter(collection):
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "alpha"), _chunk("b::0", "b", 0, "beta")],
                  [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], indexed_at=100.0)
    res = query(collection, [1.0, 0.0, 0.0], top_k=10, source_id="b")
    assert {r.chunk_index for r in res} and all("beta" == r.text for r in res)

def test_list_sources_and_latest(collection):
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "x")], [[1.0, 0.0, 0.0]], indexed_at=100.0)
    upsert_chunks(collection, [_chunk("b::0", "b", 0, "y")], [[1.0, 0.0, 0.0]], indexed_at=200.0)
    infos = {s.source_id: s for s in list_sources(collection)}
    assert infos["a"].chunk_count == 1
    assert latest_source_id(collection) == "b"

def test_duplicate_filename_overwrites(collection):
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "old")], [[1.0, 0.0, 0.0]], indexed_at=1.0)
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "new")], [[1.0, 0.0, 0.0]], indexed_at=2.0)
    assert count(collection) == 1
    assert query(collection, [1.0, 0.0, 0.0], top_k=1)[0].text == "new"

def test_delete_source(collection):
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "x"), _chunk("b::0", "b", 0, "y")],
                  [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], indexed_at=1.0)
    delete_source(collection, "a")
    assert {s.source_id for s in list_sources(collection)} == {"b"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_vector_store.py -q`
Expected: FAIL — import error.

- [ ] **Step 3: Implement `src/vector_store.py`**

`get_client` → `chromadb.PersistentClient(path=persist_dir)`. `get_or_create_collection` → `client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`. `upsert_chunks(collection, chunks, embeddings, indexed_at=0.0, source_file="")` → `collection.upsert(ids=[c.id], embeddings=..., documents=[c.text], metadatas=[{"source_id": c.source_id, "source_file": source_file, "chunk_index": c.chunk_index, "start": c.start, "end": c.end, "indexed_at": indexed_at}])`. `query` → `collection.query(query_embeddings=[vec], n_results=top_k, where={"source_id": source_id} if source_id else None)`, map to `RetrievedChunk` using distance as `score`. `list_sources` → `collection.get(include=["metadatas"])`, aggregate by `source_id` (chunk_count = number of metadatas, source_file = first seen, indexed_at = max). `latest_source_id` → argmax `indexed_at`. `delete_source` → `collection.delete(where={"source_id": source_id})`. Wrap failures in `VectorStoreError`.

- [ ] **Step 4: Run tests and verify they pass**

Run: `./venv/bin/python -m pytest tests/test_vector_store.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/vector_store.py tests/test_vector_store.py
git commit -m "feat: add persistent ChromaDB vector store operations"
```

---

### Task 6: `src/indexer.py` — transcript → time-aware chunks → index

**Files:**
- Create: `src/indexer.py`, `tests/test_indexer.py`

**Interfaces:**
- Consumes: `models.Transcript`, `models.TextChunk`, `embeddings.embed_texts`, `vector_store.upsert_chunks`, `config.Settings`.
- Produces: `chunk_transcript(t, target_chars=1200, overlap_chars=200) -> list[TextChunk]`; `index_transcript(t, collection, client, settings, indexed_at=None, progress=None) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_indexer.py`:
```python
from src.models import TranscriptSegment
from src.indexer import chunk_transcript, index_transcript
from src.vector_store import get_client, get_or_create_collection, count

def test_windows_preserve_time_span_and_ids(make_transcript):
    segs = [TranscriptSegment(index=i, start=i * 1.0, end=i * 1.0 + 1.0, text="x" * 5) for i in range(20)]
    t = make_transcript(segments=segs)
    chunks = chunk_transcript(t, target_chars=20, overlap_chars=5)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 20.0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.text.strip() for c in chunks)

def test_long_segment_is_split_and_time_preserved(make_transcript):
    segs = [TranscriptSegment(index=0, start=3.0, end=9.0, text="y" * 100)]
    chunks = chunk_transcript(make_transcript(segments=segs), target_chars=20, overlap_chars=5)
    assert len(chunks) > 1
    assert all(c.start == 3.0 and c.end == 9.0 for c in chunks)
    assert sum(len(c.text) for c in chunks) >= 100  # no text lost

def test_index_transcript_uses_embeddings(make_transcript, tmp_path):
    from types import SimpleNamespace
    class FakeEmb:
        def create(self, model, input): return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])
    client = SimpleNamespace(embeddings=FakeEmb())
    coll = get_or_create_collection(get_client(str(tmp_path / "c")), "t")
    n = index_transcript(make_transcript(), coll, client, settings=None)
    assert n == count(coll) and n > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_indexer.py -q`
Expected: FAIL — import error.

- [ ] **Step 3: Implement `src/indexer.py`**

`chunk_transcript` algorithm: walk segments; accumulate text until `len(buffer) >= target_chars`, then emit a `TextChunk(id=f"{source_id}::{n}", chunk_index=n, start=buf_start, end=buf_end, text=buffer)`; start the next buffer with trailing `overlap_chars` characters, inheriting the start time of the segment the overlap came from. If a single segment's text exceeds `target_chars`, split it into fixed `target_chars - overlap_chars` pieces, each inheriting that segment's full `[start, end]`. `index_transcript`: build chunks → `embed_texts(chunks, client, model=getattr(settings, "embedding_model", "text-embedding-3-small"), batch_size=getattr(settings, "embed_batch_size", 100))` → `upsert_chunks(..., indexed_at=time.time() or provided, source_file=t.source_file)` → return count. Use `getattr` defaults so `settings=None` and partial settings objects both work.

- [ ] **Step 4: Run tests and verify they pass**

Run: `./venv/bin/python -m pytest tests/test_indexer.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/indexer.py tests/test_indexer.py
git commit -m "feat: add transcript chunking and indexing pipeline"
```

---

### Task 7: `src/transcribe.py` — whisper-1 transcription with offset merge

**Files:**
- Create: `src/transcribe.py`, `tests/test_transcribe.py`

**Interfaces:**
- Consumes: `audio.AudioChunk`, `models.TranscriptSegment`.
- Produces: `TranscriptionError`; `transcribe_chunk(path, client, model) -> list[dict]`; `transcribe_chunks(chunks, client, model, progress=None) -> list[TranscriptSegment]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_transcribe.py`:
```python
from types import SimpleNamespace
from pathlib import Path
import pytest
from src.audio import AudioChunk
from src.transcribe import transcribe_chunks, transcribe_chunk, TranscriptionError

class FakeTranscriptions:
    def __init__(self, fail=False): self.fail = fail
    def create(self, file, model, response_format, timestamp_granularities):
        if self.fail: raise RuntimeError("boom")
        return SimpleNamespace(segments=[
            SimpleNamespace(start=0.0, end=2.0, text="a"),
            SimpleNamespace(start=2.0, end=4.0, text="b")])

class FakeClient:
    def __init__(self, fail=False): self.audio = SimpleNamespace(transcriptions=FakeTranscriptions(fail))

def _chunk(i, start, end): return AudioChunk(index=i, start=start, end=end, path=Path("/tmp/x.mp3"))

def test_local_segments():
    segs = transcribe_chunk(Path("/tmp/x.mp3"), FakeClient(), "whisper-1")
    assert [(s["start"], s["end"], s["text"]) for s in segs] == [(0.0, 2.0, "a"), (2.0, 4.0, "b")]

def test_offsets_and_renumber_across_chunks():
    segs = transcribe_chunks([_chunk(0, 0.0, 5.0), _chunk(1, 5.0, 10.0)], FakeClient(), "whisper-1")
    assert [round(s.start, 1) for s in segs] == [0.0, 2.0, 5.0, 7.0]
    assert [s.index for s in segs] == [0, 1, 2, 3]

def test_boundary_segment_offsets_correctly():
    segs = transcribe_chunks([_chunk(1, 300.0, 305.0)], FakeClient(), "whisper-1")
    assert segs[0].start == 300.0 and segs[-1].end == 304.0

def test_progress_callback_reports():
    seen = []
    transcribe_chunks([_chunk(0, 0, 5), _chunk(1, 5, 10)], FakeClient(), "whisper-1",
                      progress=lambda cur, total: seen.append((cur, total)))
    assert seen == [(1, 2), (2, 2)]

def test_api_error_wrapped():
    with pytest.raises(TranscriptionError):
        transcribe_chunks([_chunk(0, 0, 5)], FakeClient(fail=True), "whisper-1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_transcribe.py -q`
Expected: FAIL — import error.

- [ ] **Step 3: Implement `src/transcribe.py`**

`transcribe_chunk` calls `client.audio.transcriptions.create(file=open(path,"rb"), model=model, response_format="verbose_json", timestamp_granularities=["segment"])`, returns `[{"start": s.start, "end": s.end, "text": s.text} for s in resp.segments]`. `transcribe_chunks` iterates chunks, offsets each local segment by `chunk.start`, builds `TranscriptSegment` with a running `index`, calls `progress(i+1, total)`. Wrap per-chunk errors with a small retry (2 retries, `time.sleep` backoff) then raise `TranscriptionError`. Guard against `resp.segments` being `None` (silent audio) → treat as empty list.

- [ ] **Step 4: Run tests and verify they pass**

Run: `./venv/bin/python -m pytest tests/test_transcribe.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/transcribe.py tests/test_transcribe.py
git commit -m "feat: add whisper-1 transcription with offset merge"
```

---

### Task 8: `src/rag.py` — retrieval + GPT-4o answers with citations

**Files:**
- Create: `src/rag.py`, `tests/test_rag.py`

**Interfaces:**
- Consumes: `models.RetrievedChunk`, `embeddings.embed_query`, `vector_store.query`, `config.Settings`, an injectable LLM with `.invoke(messages) -> obj.content`.
- Produces: `format_timestamp(seconds) -> str`; `build_context(retrieved) -> str`; `build_messages(question, retrieved) -> list[dict]`; `SYSTEM_PROMPT`; `answer_question(question, collection, client, settings, source_id=None, top_k=None, llm=None) -> tuple[str, list[RetrievedChunk]]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_rag.py`:
```python
from types import SimpleNamespace
import pytest
from src.models import RetrievedChunk, TextChunk
from src.rag import format_timestamp, build_context, build_messages, answer_question
from src.vector_store import get_client, get_or_create_collection, upsert_chunks

def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(63) == "01:03"
    assert format_timestamp(3724) == "1:02:04"

def test_build_context_has_timestamps():
    ctx = build_context([RetrievedChunk(text="revenue up", start=12.0, end=45.0, score=0.1, chunk_index=0)])
    assert "[00:12-00:45]" in ctx and "revenue up" in ctx

def test_build_messages_system_and_context():
    msgs = build_messages("How was revenue?", [RetrievedChunk(text="x", start=0, end=1, score=0, chunk_index=0)])
    assert msgs[0]["role"] == "system"
    assert "MM:SS" in msgs[0]["content"] or "mm:ss" in msgs[0]["content"].lower()
    assert "How was revenue?" in msgs[-1]["content"]

@pytest.fixture
def collection(tmp_path):
    return get_or_create_collection(get_client(str(tmp_path / "c")), "t")

def test_answer_question_stub_llm(collection, make_transcript, tmp_path):
    from src.indexer import index_transcript
    from types import SimpleNamespace
    class FakeEmb:
        def create(self, model, input): return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])
    client = SimpleNamespace(embeddings=FakeEmb())
    index_transcript(make_transcript(), collection, client, settings=None)
    llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="Revenue grew [00:01]"))
    answer, retrieved = answer_question("revenue?", collection, client, settings=SimpleNamespace(embedding_model="m", rag_top_k=6), llm=llm)
    assert "Revenue grew" in answer
    assert retrieved  # citations returned for the UI
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_rag.py -q`
Expected: FAIL — import error.

- [ ] **Step 3: Implement `src/rag.py`**

`format_timestamp`: `H:MM:SS` when >= 3600 s else `MM:SS`. `build_context`: join `f"[{format_timestamp(c.start)}-{format_timestamp(c.end)}] {c.text}"` with newlines. `SYSTEM_PROMPT` instructs: answer only from context; cite timestamps as `[MM:SS]`; if context is insufficient, say so. `build_messages`: `[{"role":"system","content":SYSTEM_PROMPT}, {"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}"}]`. `answer_question`: `embed_query` → `vector_store.query` → if no retrieved, return a graceful "no relevant context" message with `[]`; else `llm.invoke(build_messages(...))`; build the default `ChatOpenAI(model=settings.chat_model, temperature=0)` only when `llm is None`.

- [ ] **Step 4: Run tests and verify they pass**

Run: `./venv/bin/python -m pytest tests/test_rag.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/rag.py tests/test_rag.py
git commit -m "feat: add RAG question answering with timestamped citations"
```

---

### Task 9: `src/pipeline.py` — end-to-end orchestration

**Files:**
- Create: `src/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `audio.*`, `transcribe.*`, `transcript_io.*`, `indexer.index_transcript`, `models.Transcript/IngestResult`, `config.Settings`.
- Produces: `ingest_audio(audio_path, collection, client, settings, limit_seconds=None, progress=None) -> IngestResult`; `ingest_transcript(transcript, collection, client, settings, progress=None) -> int`.

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:
```python
from types import SimpleNamespace
from src.pipeline import ingest_audio, ingest_transcript
from src.vector_store import get_client, get_or_create_collection, count

class FakeEmb:
    def create(self, model, input): return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])

def _client(): return SimpleNamespace(embeddings=FakeEmb())

def test_ingest_transcript_saves_and_indexes(make_transcript, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    coll = get_or_create_collection(get_client(str(tmp_path / "c")), "t")
    n = ingest_transcript(make_transcript(), coll, _client(), settings=None)
    assert n == count(coll) and n > 0
    assert (tmp_path / "transcripts" / "pypl.json").exists()

def test_ingest_audio_uses_split_and_transcribe(monkeypatch, tmp_path):
    from src import pipeline, audio
    from src.models import TranscriptSegment
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "split_audio", lambda *a, **k: [], raising=True)
    monkeypatch.setattr(pipeline, "transcribe_chunks",
                        lambda chunks, client, model, progress=None: [TranscriptSegment(index=0, start=0.0, end=1.0, text="hi")])
    coll = get_or_create_collection(get_client(str(tmp_path / "c")), "t")
    result = ingest_audio("call.mp3", coll, _client(), settings=SimpleNamespace(
        transcription_model="whisper-1", audio_chunk_seconds=300))
    assert result.transcript.source_id == "call"
    assert result.chunk_count == count(coll)
    assert (tmp_path / "transcripts" / "call.srt").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: FAIL — import error.

- [ ] **Step 3: Implement `src/pipeline.py`**

`ingest_transcript`: `n = index_transcript(t, collection, client, settings)`; `save_transcript(t, out_dir="transcripts")`; also write the `.srt` (`(out_dir/f"{source_id}.srt").write_text(transcript_to_srt(t))`); return `n`. `ingest_audio`: `chunks = split_audio(audio_path, chunk_seconds=settings.audio_chunk_seconds, limit_seconds=limit_seconds)`; `segments = transcribe_chunks(chunks, client, settings.transcription_model, progress)`; `duration = chunks[-1].end if chunks else (segments[-1].end if segments else 0.0)`; assemble `Transcript(source_id=slugify(Path(audio_path).name), source_file=Path(audio_path).name, duration_seconds=duration, transcription_model=settings.transcription_model, created_at=utcnow isoformat, segments=segments)`; `cleanup_chunks(chunks)`; then `ingest_transcript(...)`; return `IngestResult`. Re-export `split_audio`/`transcribe_chunks` at module level (imported names) so tests can monkeypatch them.

- [ ] **Step 4: Run tests and verify they pass**

Run: `./venv/bin/python -m pytest tests/test_pipeline.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add end-to-end ingestion orchestration"
```

---

### Task 10: `app.py` — Streamlit UI

**Files:**
- Create: `app.py`, `tests/test_app.py`

**Interfaces:**
- Consumes: everything above; `config.load_settings`, `vector_store.*`, `pipeline.*`, `rag.answer_question`.
- Produces: a runnable `streamlit run app.py` experience (no importable API).

- [ ] **Step 1: Write the failing test (headless app boot)**

`tests/test_app.py`:
```python
from streamlit.testing.v1 import AppTest

def test_app_boots_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    # sidebar should surface the missing-key guidance
    assert any("OPENAI_API_KEY" in (w.value or "") for w in at.sidebar.markdown)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_app.py -q`
Expected: FAIL — `app.py` not found / no workdir file.

- [ ] **Step 3: Implement `app.py`**

Structure:
- `st.set_page_config(page_title="StreamPulse-RAG", layout="wide")`.
- `settings = load_settings()`; `collection = get_or_create_collection(get_client(settings.chroma_dir), settings.collection_name)`; cache the collection via `@st.cache_resource`.
- **Sidebar:** API-key status. If `not settings.has_api_key`, show `st.sidebar.error("OPENAI_API_KEY is not set. Add it to .env and restart.")` and set a guard flag that disables network buttons (use `st.button(..., disabled=...)`). Chat-model select. Source selector: `st.sidebar.selectbox` over `list_sources()` formatted `f"{s.source_file} ({s.source_id})"`, default index = the `latest_source_id`. Scope toggle `st.sidebar.radio("Search scope", ["Selected call", "All calls"])`. Source list with delete buttons.
- **Tab "Transcribe Audio":** `st.file_uploader(type=["mp3","wav"])`; write bytes to a temp file; `st.audio`; "Transcribe" button → `pipeline.ingest_audio(..., progress)` updating `st.progress`; store `result.transcript` in `st.session_state`; preview first 20 segments; `st.download_button` for JSON (`model_dump_json`) and SRT.
- **Tab "Upload Transcript":** `st.file_uploader(type=["json","srt"])` → `parse_transcript_file`; preview; "Index" → `ingest_transcript`.
- **Chat:** `st.chat_input`; on submit, `answer_question(question, collection, client, settings, source_id=(selected or None if All calls), llm=ChatOpenAI(...))`; render `st.chat_message` with the answer; expander listing retrieved chunks as `[mm:ss-mm:ss] text`.
- Guard all API-calling actions behind `settings.has_api_key`; show `st.warning` instead of crashing.
- Read the missing-key message from an `st.sidebar.markdown`/`error` whose text contains the literal `OPENAI_API_KEY` (so the test can find it).

- [ ] **Step 4: Run test and manual boot**

Run: `./venv/bin/python -m pytest tests/test_app.py -q`
Expected: `1 passed`.
Then: `./venv/bin/streamlit run app.py --server.headless true` (start, confirm it serves, then stop).

- [ ] **Step 5: Commit**

```bash
git add app.py tests/test_app.py
git commit -m "feat: add Streamlit UI for upload, transcription, and Q&A"
```

---

### Task 11: End-to-end excerpt test, docs, and full sanity run

**Files:**
- Create: `tests/test_e2e_excerpt.py`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: the full pipeline; real `OPENAI_API_KEY`.
- Produces: a key-gated integration test + updated run docs.

- [ ] **Step 1: Write the key-gated e2e test**

`tests/test_e2e_excerpt.py`:
```python
import os
from pathlib import Path
import pytest
from src.config import load_settings, get_openai_client
from src.pipeline import ingest_audio
from src.rag import answer_question
from src.vector_store import get_client, get_or_create_collection, count

AUDIO = Path("data/PYPL_Q4_2025_Earnings_Call.mp3")

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY") or not AUDIO.exists(),
    reason="requires OPENAI_API_KEY and data/PYPL_Q4_2025_Earnings_Call.mp3",
)

def test_pypl_excerpt_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load_settings()
    client = get_openai_client(s.openai_api_key)
    coll = get_or_create_collection(get_client(str(tmp_path / "chroma")), "e2e")
    result = ingest_audio(AUDIO, coll, client, s, limit_seconds=120)
    assert result.chunk_count > 0
    assert len(result.transcript.segments) > 0
    answer, retrieved = answer_question("What did management say about revenue?", coll, client, s)
    assert answer.strip()
    assert retrieved
```

- [ ] **Step 2: Run the full suite (keyless) to confirm it skips cleanly**

Run: `./venv/bin/python -m pytest -q`
Expected: all pass; `1 skipped` (or `2 skipped`) for key-gated tests; 0 failures.

- [ ] **Step 3: Run the real excerpt test (after the key is added to `.env`)**

Run: `./venv/bin/python -m pytest tests/test_e2e_excerpt.py -v`
Expected: PASS; the transcript, `transcripts/pypl_q4_2025_earnings_call.json/.srt`, and a non-zero Chroma chunk count are produced. *(If the key is not yet set, this reports SKIPPED — report that honestly rather than claiming success.)*

- [ ] **Step 4: Update `AGENTS.md`**

Add a "Module map" section mirroring spec §6 and a "Run & test" section:
```
source venv/bin/activate
pytest -q                          # keyless sanity suite
pytest tests/test_e2e_excerpt.py   # real excerpt test (needs OPENAI_API_KEY)
streamlit run app.py
```
Update the tech-stack line to note whisper-1 and no speaker labels.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e_excerpt.py AGENTS.md
git commit -m "test: add key-gated PYPL excerpt e2e and update docs"
```

---

## Self-Review

**1. Spec coverage**
- §7 data contracts → Task 1. §8.2 audio → Task 2. §8.4 transcript_io → Task 3. §8.5 embeddings → Task 4. §8.6 vector_store (incl. `latest_source_id`, multi-source, delete) → Task 5. §8.7 indexer → Task 6. §8.3 transcribe → Task 7. §8.8 rag → Task 8. §8.9 pipeline → Task 9. §8.10 UI (source selector, scope toggle, both tabs, downloads) → Task 10. §12 e2e → Task 11. §13 deliverables (pytest, `.env.example`, `.gitignore`, `AGENTS.md`) → Tasks 1 & 11.

**2. Step scan** — every step names one action with a checkable result; implementation steps give signatures + only the algorithms the tests don't determine (chunker, SRT, timestamp formatting). No "handle edge cases" filler.

**3. Type consistency** — `upsert_chunks(collection, chunks, embeddings, indexed_at=0.0, source_file="")` is called identically in Tasks 5, 6; `transcribe_chunks(chunks, client, model, progress)` consistent in Tasks 7, 9; `answer_question(..., source_id, top_k, llm)` consistent in Tasks 8, 10. `TextChunk.id` format `f"{source_id}::{chunk_index}"` used in Tasks 5, 6. `index_transcript` reads settings defensively (`getattr`) so `settings=None` (Task 6) and partial settings (Task 9) both work.

**4. Review Focus** — silent/short audio → Task 2 (`test_short_audio_single_chunk`) + Task 7 (`resp.segments is None` guard); chunk-boundary offsets → Task 7 (`test_boundary_segment_offsets_correctly`); long-segment split → Task 6 (`test_long_segment_is_split_and_time_preserved`); duplicate filename → Task 5 (`test_duplicate_filename_overwrites`); malformed upload → Task 3 (`test_malformed_json_raises`, `test_unknown_extension_raises`).

**5. Proportion** — plan carries signatures, test assertions, and the few non-obvious algorithms; no full implementations transcribed.
