"""Key-gated end-to-end excerpt test against the real PYPL earnings call.

This is the only test that exercises the full stack against live OpenAI APIs:
whisper-1 transcribes the first 120 seconds of the real MP3, the transcript is
chunked/embedded/indexed into a throwaway ChromaDB, and gpt-4o answers a
question grounded in the retrieved, timestamped context.

It is skipped unless an API key is configured in ``.env`` *and* the real audio
file is present, so the default keyless suite stays fast and offline.
"""

from pathlib import Path

import pytest

from src.config import get_openai_client, load_settings
from src.pipeline import ingest_audio
from src.rag import answer_question
from src.vector_store import get_client, get_or_create_collection

# Absolute path: the test chdir()s into tmp_path, so a relative path would break.
AUDIO = Path(__file__).resolve().parent.parent / "data" / "PYPL_Q4_2025_Earnings_Call.mp3"

# load_settings() reads .env, so the gate honors a key that is only in .env
# (not necessarily exported into the pytest process). Never print the key.
pytestmark = pytest.mark.skipif(
    not load_settings().has_api_key or not AUDIO.exists(),
    reason="requires OPENAI_API_KEY (in .env) and data/PYPL_Q4_2025_Earnings_Call.mp3",
)


def test_pypl_excerpt_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load_settings()
    client = get_openai_client(s.openai_api_key)
    coll = get_or_create_collection(get_client(str(tmp_path / "chroma")), "e2e")
    result = ingest_audio(AUDIO, coll, client, s, limit_seconds=120)
    assert result.chunk_count > 0
    assert len(result.transcript.segments) > 0
    answer, retrieved = answer_question(
        "What did management say about revenue?", coll, client, s
    )
    assert answer.strip()
    assert retrieved
