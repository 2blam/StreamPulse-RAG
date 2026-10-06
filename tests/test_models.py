from src.models import Transcript, TranscriptSegment


def test_transcript_json_roundtrip():
    t = Transcript(source_id="x", source_file="x.mp3", duration_seconds=10.0,
                   transcription_model="whisper-1", created_at="2026-01-01T00:00:00Z",
                   segments=[TranscriptSegment(index=0, start=0.0, end=1.0, text="hi")])
    again = Transcript.model_validate_json(t.model_dump_json())
    assert again.segments[0].text == "hi"
    assert again.schema_version == "1.0"
