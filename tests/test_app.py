"""Headless smoke test for the Streamlit UI.

``AppTest`` runs ``app.py`` in-process and surfaces any uncaught exception as
``at.exception``. The app must boot cleanly even when no OpenAI key is
available, showing the missing-key guidance instead of crashing.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

import src.config

# AppTest resolves relative paths against the calling file (tests/), so anchor
# the entrypoint at the repository root explicitly.
APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


def test_app_boots_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # A developer's local .env would otherwise be read back into the environment
    # by load_settings(), masking the missing-key state under test.
    monkeypatch.setattr(src.config, "load_dotenv", lambda *args, **kwargs: None)

    at = AppTest.from_file(str(APP_PATH), default_timeout=30).run()

    assert not at.exception
    # The sidebar must surface the missing-key guidance via st.error.
    assert any("OPENAI_API_KEY" in (w.value or "") for w in at.sidebar.error)
