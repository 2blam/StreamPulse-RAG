# AGENTS.md — StreamPulse-RAG

> Guidance for AI coding agents (and humans) working in this repository.

## 1. Project Context

**StreamPulse-RAG** is a local, multimodal Retrieval-Augmented Generation (RAG) engine for
**FinTech earnings calls**. Users upload `.mp3` audio recordings of earnings calls
(e.g., PayPal, Adyen, Block) directly through the UI. The system transforms long-form
audio into **timestamped, C-suite intelligence briefings** backed by searchable vector
embeddings.

The goal is to give analysts and investors a fast, conversational way to interrogate
earnings calls: ask a question, get an answer grounded in the actual call, with
timestamps pointing back to the source audio.

## 2. Core Architecture

```
File Upload (Streamlit MP3)
        │
        ▼
Audio Processing & Chunking
        │
        ▼
Vector Store (ChromaDB)
        │
        ▼
Multimodal RAG Engine (GPT-4o)
        │
        ▼
Streamlit Dashboard
```

| Stage | Responsibility |
|-------|----------------|
| **File Upload** | Streamlit UI accepts local `.mp3` earnings call files. |
| **Audio Processing & Chunking** | Transcribe audio (timestamped), then chunk the transcript with time metadata. |
| **Vector Store** | Embed chunks with `text-embedding-3-small` and persist to ChromaDB. |
| **Multimodal RAG Engine** | GPT-4o answers questions using retrieved, timestamped context. |
| **Streamlit Dashboard** | Interactive Q&A, metric extraction, and visualization. |

## 3. Primary Metrics to Analyze

Every briefing should aim to surface the following FinTech-specific signals:

- **Revenue Growth** — YoY / QoQ revenue, guidance vs. consensus.
- **Take Rate Guidance** — net take rate, monetization trends, pricing changes.
- **Cross-border Volume** — TPV growth, cross-border mix, FX impact.
- **Margin Expansion** — operating margin, transaction margin, opex leverage.
- **C-Suite Q&A Sentiment** — tone and confidence of executives during analyst Q&A.
- **Key Takeaways** — grounded, timestamped summary points per call.

## 4. Data Source

- **Source:** User-uploaded **local `.mp3`** earnings call files.
- **No external crawling** of transcripts or audio; all inputs are provided by the user.
- Audio/transcript artifacts are **git-ignored** and never committed.

## 5. Tech Stack

- **Language:** Python 3.13 (project venv; Python 3.12+ supported)
- **UI:** Streamlit
- **LLM:** OpenAI GPT-4o (`OPENAI_MODEL_NAME=gpt-4o`)
- **Transcription:** OpenAI `whisper-1` (`verbose_json`, segment timestamps) — **no speaker labels** (diarization intentionally omitted)
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector Store:** ChromaDB (persisted to `chroma_db/`)
- **Orchestration:** LangChain (`langchain-openai`, `langchain-community`)
- **Audio:** `pydub` (requires `ffmpeg` on the host PATH)
- **Models/Config:** Pydantic, `python-dotenv`
- **Visualization:** Plotly

## 6. Module Map

Mirrors the design spec's module layout (spec §6):

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
docs/superpowers/specs/      # design spec & plans
chroma_db/                   # runtime vector store (gitignored)
transcripts/                 # saved transcript artifacts (runtime; JSON/SRT gitignored)
```

## 7. Conventions

- Keep secrets in `.env` (never committed). Copy from `.env.example`.
- Store all OpenAI keys/config via environment variables.
- Prefer small, single-responsibility modules under `src/`.
- All media and vector-store artifacts are local-only and git-ignored.

## 8. Local Development

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Configure environment
cp .env.example .env        # then set OPENAI_API_KEY

# 3. Run the smoke test
python smoke_test.py

# 4. Launch the app (once implemented)
streamlit run app.py
```

## 9. Run & Test

```bash
source venv/bin/activate
pytest -q                          # keyless sanity suite
pytest tests/test_e2e_excerpt.py   # real excerpt test (needs OPENAI_API_KEY)
streamlit run app.py
```

`pytest tests/test_e2e_excerpt.py` transcribes the first 120 s of
`data/PYPL_Q4_2025_Earnings_Call.mp3` with `whisper-1`, indexes it into a
throwaway ChromaDB, and answers a question with `gpt-4o` — it makes real API
calls and is skipped unless `OPENAI_API_KEY` is set in `.env`.

## 10. Agent Operating Notes

- **Never commit** `venv/`, `.env`, `*.mp3`, `*.wav`, `chroma_db/`, or `__pycache__/`.
- Treat uploaded audio as sensitive: do not exfiltrate or log raw transcripts.
- When adding dependencies, update `requirements.txt`.
- Verify changes with `smoke_test.py` before claiming success.
