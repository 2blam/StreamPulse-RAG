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
