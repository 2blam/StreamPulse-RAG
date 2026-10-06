from types import SimpleNamespace

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
