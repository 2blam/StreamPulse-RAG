from types import SimpleNamespace

import pytest

from src.pipeline import ingest_audio, ingest_transcript
from src.vector_store import get_client, get_or_create_collection, count


class FakeEmb:
    def create(self, model, input):
        return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])


def _client():
    return SimpleNamespace(embeddings=FakeEmb())


def test_ingest_transcript_saves_and_indexes(make_transcript, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    coll = get_or_create_collection(get_client(str(tmp_path / "c")), "test")
    n = ingest_transcript(make_transcript(), coll, _client(), settings=None)
    assert n == count(coll) and n > 0
    assert (tmp_path / "transcripts" / "pypl.json").exists()
    assert (tmp_path / "transcripts" / "pypl.srt").exists()


def test_ingest_audio_uses_split_and_transcribe(monkeypatch, tmp_path):
    from src import pipeline
    from src.models import TranscriptSegment

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "split_audio", lambda *a, **k: [], raising=True)
    monkeypatch.setattr(
        pipeline,
        "transcribe_chunks",
        lambda chunks, client, model, progress=None: [
            TranscriptSegment(index=0, start=0.0, end=1.0, text="hi")
        ],
    )
    coll = get_or_create_collection(get_client(str(tmp_path / "c")), "test")
    result = ingest_audio(
        "call.mp3",
        coll,
        _client(),
        settings=SimpleNamespace(transcription_model="whisper-1", audio_chunk_seconds=300),
    )
    assert result.transcript.source_id == "call"
    assert result.transcript.duration_seconds == 1.0
    assert result.chunk_count == count(coll)
    assert (tmp_path / "transcripts" / "call.srt").exists()
    assert (tmp_path / "transcripts" / "call.json").exists()


def test_ingest_audio_cleans_up_on_transcription_failure(monkeypatch, tmp_path):
    from src import pipeline
    from src.audio import AudioChunk
    from src.transcribe import TranscriptionError

    monkeypatch.chdir(tmp_path)
    chunk = AudioChunk(index=0, start=0.0, end=1.0, path=tmp_path / "chunk_0000.mp3")
    calls = []
    monkeypatch.setattr(pipeline, "split_audio", lambda *a, **k: [chunk], raising=True)
    monkeypatch.setattr(pipeline, "cleanup_chunks", lambda chunks: calls.append(chunks))

    def _boom(*args, **kwargs):
        raise TranscriptionError("boom")

    monkeypatch.setattr(pipeline, "transcribe_chunks", _boom)

    coll = get_or_create_collection(get_client(str(tmp_path / "c")), "test")
    with pytest.raises(TranscriptionError):
        ingest_audio(
            "call.mp3",
            coll,
            _client(),
            settings=SimpleNamespace(transcription_model="whisper-1", audio_chunk_seconds=300),
        )
    assert len(calls) == 1
    assert calls[0] == [chunk]


def test_ingest_audio_uses_original_filename_as_source_id(monkeypatch, tmp_path):
    from src import pipeline
    from src.models import TranscriptSegment

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "split_audio", lambda *a, **k: [], raising=True)
    monkeypatch.setattr(
        pipeline,
        "transcribe_chunks",
        lambda chunks, client, model, progress=None: [
            TranscriptSegment(index=0, start=0.0, end=1.0, text="hi")
        ],
    )
    coll = get_or_create_collection(get_client(str(tmp_path / "c")), "test")
    result = ingest_audio(
        "/tmp/tmp3ezbz488.mp3",  # temp path written by the Streamlit uploader
        coll,
        _client(),
        settings=SimpleNamespace(transcription_model="whisper-1", audio_chunk_seconds=300),
        source_name="PYPL_Q4_2025_Earnings_Call.mp3",
    )
    assert result.transcript.source_id == "pypl_q4_2025_earnings_call"
    assert result.transcript.source_file == "PYPL_Q4_2025_Earnings_Call.mp3"
    assert (tmp_path / "transcripts" / "pypl_q4_2025_earnings_call.json").exists()
