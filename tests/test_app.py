"""Headless smoke tests for the Streamlit UI.

``AppTest`` runs ``app.py`` in-process and surfaces any uncaught exception as
``at.exception``. The app must boot cleanly even when no OpenAI key is
available, showing the missing-key guidance instead of crashing, and must
surface (rather than leak) runtime failures from the API-calling paths.
"""

from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

import src.config
import src.rag

# AppTest resolves relative paths against the calling file (tests/), so anchor
# the entrypoint at the repository root explicitly.
APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def _isolate_env(monkeypatch, chroma_dir: str) -> None:
    """Point the app at a throwaway store and keep tests independent of .env."""
    monkeypatch.setenv("CHROMA_DIR", chroma_dir)
    # A developer's local .env would otherwise be read back into the environment
    # by load_settings(), masking the state under test.
    monkeypatch.setattr(src.config, "load_dotenv", lambda *args, **kwargs: None)


def _seed_collection(chroma_dir: str) -> None:
    """Index one small transcript so the app has a source to chat about."""
    from src.indexer import index_transcript
    from src.models import Transcript, TranscriptSegment
    from src.vector_store import get_client, get_or_create_collection

    fake_client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda model, input: SimpleNamespace(
                data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input]
            )
        )
    )
    collection = get_or_create_collection(get_client(chroma_dir), "earnings_calls")
    transcript = Transcript(
        source_id="pypl",
        source_file="PYPL.mp3",
        duration_seconds=1.5,
        transcription_model="whisper-1",
        created_at="2026-01-01T00:00:00Z",
        segments=[TranscriptSegment(index=0, start=0.0, end=1.5, text="revenue grew")],
    )
    index_transcript(transcript, collection, fake_client, settings=None)


def test_app_boots_without_key(monkeypatch, tmp_path):
    _isolate_env(monkeypatch, str(tmp_path / "chroma"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not at.exception
    # The sidebar must surface the missing-key guidance via st.error.
    assert any("OPENAI_API_KEY" in (w.value or "") for w in at.sidebar.error)


def test_chat_error_is_surfaced(monkeypatch, tmp_path):
    chroma_dir = str(tmp_path / "chroma")
    _isolate_env(monkeypatch, chroma_dir)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy")
    _seed_collection(chroma_dir)

    def boom(*args, **kwargs):
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr(src.rag, "answer_question", boom)

    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()
    at.chat_input[0].set_value("How was revenue?").run()

    assert not at.exception
    assert any("Answering failed" in (e.value or "") for e in at.error)
