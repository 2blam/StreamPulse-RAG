# StreamPulse-RAG

**Multimodal & Streaming RAG for FinTech Earnings Calls — turn long MP3 earnings calls into timestamped, searchable, C-suite intelligence briefings.**

Upload an earnings-call recording (PayPal, Adyen, Block, …) and ask questions in plain English. StreamPulse-RAG transcribes the audio, indexes it into a local vector store, and answers with GPT-4o — every answer grounded in the call and cited with the exact `[mm:ss]` timestamps it came from.

![StreamPulse-RAG Streamlit dashboard: an indexed earnings call, a question ("What did management say about revenue growth?"), and a grounded GPT-4o answer citing [34:07-35:24] and [25:52-27:29]](docs/assets/streamlit-dashboard.png)

**The problem it solves:** earnings calls are 60–90 minutes of unstructured audio. Analysts either listen to the whole thing or skim an unaudited third-party transcript. StreamPulse-RAG turns the raw audio into a private, queryable knowledge base — so "what did management guide for cross-border volume?" returns an answer with the source timestamps, in seconds.

---

## Key Features

- **Local MP3 upload** — drop an earnings-call file straight into the Streamlit UI (no external crawling).
- **Timestamped transcription** — OpenAI `whisper-1` (`verbose_json`, segment-level timestamps) with per-chunk offset merging across the whole call.
- **Reusable transcript artifacts** — download the transcript as **JSON + SRT**, and re-upload it later to **skip re-transcription** entirely.
- **Time-aware chunking** — transcript segments are merged into overlapping text windows that preserve their `[start, end]` span, so citations point back at the audio.
- **Local vector store** — chunks embedded with `text-embedding-3-small` and persisted to **ChromaDB** (cosine space).
- **Multi-source library** — index many calls side by side; a sidebar **source selector** (`filename (source_id)`) plus a **Search scope** toggle (*Selected call* / *All calls*).
- **Grounded Q&A with citations** — GPT-4o answers only from retrieved context and cites `[mm:ss]` timestamps, with expandable source chunks.
- **FinTech-aware analysis** — designed to surface Revenue Growth, Take Rate Guidance, Cross-border Volume, Margin Expansion, and Key Takeaways.
- **Tested** — 53 offline unit/integration tests plus a key-gated end-to-end test against a real call.

> **Note:** speaker labels (diarization) are intentionally **out of scope** — transcripts carry timestamps but no speaker attribution.

---

## Architecture & Pipeline

```
                         ┌──────────────────────────┐
                         │   MP3 Upload (Streamlit) │
                         └────────────┬─────────────┘
                                      │
                                      ▼
              Audio Processing & Chunking  (pydub → 5-min, mono 16 kHz)
                                      │
                                      ▼
              Transcription  (OpenAI whisper-1 · verbose_json · segment timestamps)
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
        Transcript Artifact (JSON + SRT)        Upload Transcript (JSON/SRT)
        downloadable / re-uploadable                     │  (skips transcription)
                     │                                 │
                     └────────────────┬────────────────┘
                                      ▼
        Time-aware Chunking → Embeddings (text-embedding-3-small)
                                      │
                                      ▼
                  Vector Store (ChromaDB · persistent · multi-source)
                                      │
                                      ▼
        Multimodal RAG Engine (GPT-4o) → answers with [mm:ss] citations
                                      │
                                      ▼
        Streamlit Dashboard (source selector · scope toggle · chat)
```

**Data flow in one line:** `File Upload → Audio Chunking → Vector Store (ChromaDB) → Multimodal RAG Engine (GPT-4o) → Streamlit Dashboard`.

---

## Prerequisites & Installation

**Prerequisites**
- **Python 3.12+** (developed and tested on 3.13)
- **ffmpeg** on your `PATH` (required by `pydub` for MP3 decoding)
- An **OpenAI API key**

**Install ffmpeg**
```bash
# macOS
brew install ffmpeg
# Debian / Ubuntu
sudo apt-get install ffmpeg
```

**Install the project**
```bash
git clone <your-repo-url>
cd StreamPulse-RAG

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> On Python 3.13, `requirements.txt` also pulls `audioop-lts` — the backport of the stdlib `audioop` module that `pydub` needs (it was removed in 3.13).

---

## Configuration

Copy the template and add your key:

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key. **Never commit this.** |
| `OPENAI_MODEL_NAME` | `gpt-4o` | Chat/answering model. |
| `TRANSCRIPTION_MODEL` | `whisper-1` | Transcription model (needs timestamps). |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model. |
| `CHROMA_DIR` | `chroma_db` | Local ChromaDB persistence directory. |
| `AUDIO_CHUNK_SECONDS` | `300` | Audio chunk length in seconds (5 minutes). |

`.env` is git-ignored. Only `.env.example` (with an empty key) is committed.

---

## Usage

### Run the app

```bash
source venv/bin/activate
streamlit run app.py
```

Open <http://localhost:8501>, then:

1. **Transcribe Audio tab** — upload an MP3/WAV → play it → **Transcribe** (progress bar) → preview the transcript → **Download JSON / SRT**. It auto-indexes on success.
2. **Upload Transcript tab** — upload a previously downloaded **JSON/SRT** to skip transcription, then **Index**.
3. **Chat** — ask questions like *"What did management say about take rate guidance?"* Answers include `[mm:ss]` citations and expandable source chunks.
4. **Sidebar** — pick the active call (source selector) and scope queries to *Selected call* or *All calls*.

### Run the tests

```bash
# Offline suite — no API calls, no key needed (fast, free)
pytest -q --ignore=tests/test_e2e_excerpt.py

# Full suite — also runs the real-API e2e when .env has a key
pytest -q

# Real end-to-end test only (paid: whisper-1 + gpt-4o, ~15s)
pytest tests/test_e2e_excerpt.py
```

> The end-to-end test transcribes the first 120 seconds of `data/PYPL_Q4_2025_Earnings_Call.mp3`, indexes it, and answers a question. It is **skipped** only when either `OPENAI_API_KEY` is absent or the audio file is missing — otherwise it **runs and incurs real API spend**.

### Cost & performance notes

- Audio is chunked at 5 minutes and re-encoded to mono 16 kHz, keeping every request well under OpenAI's 25 MB upload limit (even for very large source files).
- A full 60–90 minute call takes a few minutes to transcribe and costs more than the 120 s test excerpt.

---

## Project Structure

```
StreamPulse-RAG/
├── app.py                        # Streamlit UI (thin orchestrator)
├── smoke_test.py                 # quick environment / dependency smoke check
├── requirements.txt
├── .env.example                  # config template (no secrets)
├── src/
│   ├── config.py                 # env loading, paths, model names, constants
│   ├── models.py                 # Pydantic data contracts
│   ├── audio.py                  # pydub probe + split into timestamped 5-min chunks
│   ├── transcribe.py             # whisper-1 transcription + offset merge + progress
│   ├── transcript_io.py          # JSON save/load, SRT export, upload parsing
│   ├── embeddings.py             # text-embedding-3-small (batched)
│   ├── vector_store.py           # ChromaDB persistent client operations
│   ├── indexer.py                # transcript → time-aware chunks → embeddings → Chroma
│   ├── rag.py                    # retrieval + GPT-4o answer with timestamped citations
│   └── pipeline.py               # end-to-end orchestration + progress callbacks
├── tests/
│   ├── conftest.py               # shared fixtures
│   ├── test_audio.py             # chunking offsets, limits, error handling
│   ├── test_transcript_io.py     # JSON/SRT round-trip, slugify, malformed uploads
│   ├── test_embeddings.py        # batching
│   ├── test_vector_store.py      # upsert/query, multi-source, delete
│   ├── test_indexer.py           # windowing + time-span preservation
│   ├── test_transcribe.py        # offset merge, silent audio, retries
│   ├── test_rag.py               # timestamp formatting, prompt assembly
│   ├── test_pipeline.py          # ingestion orchestration
│   ├── test_app.py               # headless Streamlit boot
│   └── test_e2e_excerpt.py       # key-gated real-API end-to-end
├── docs/superpowers/             # design spec & implementation plan
├── docs/assets/                  # README screenshots
├── data/                         # local audio (git-ignored)
├── chroma_db/                    # runtime vector store (git-ignored)
└── transcripts/                  # saved JSON/SRT artifacts (git-ignored)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Transcription | OpenAI `whisper-1` (`verbose_json`) |
| LLM | OpenAI `gpt-4o` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (persistent, cosine) |
| Audio | pydub + ffmpeg |
| Orchestration | LangChain OpenAI (`langchain-openai`) |
| Models / config | Pydantic, python-dotenv |
| Visualization | Plotly |

## License

Add a license of your choice (e.g. MIT) before publishing.
