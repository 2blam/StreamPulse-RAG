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
