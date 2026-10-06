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
