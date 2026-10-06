import pytest
from src.transcript_io import (slugify, save_transcript, load_transcript,
                               transcript_to_srt, parse_transcript_file, TranscriptFormatError)

def test_slugify():
    assert slugify("PYPL_Q4_2025_Earnings_Call.mp3") == "pypl_q4_2025_earnings_call"
    assert slugify("$XYZ Block Inc Q4 2025 [id].mp3") == "xyz_block_inc_q4_2025_id"

def test_save_load_roundtrip(make_transcript, tmp_path):
    t = make_transcript()
    p = save_transcript(t, out_dir=tmp_path)
    assert p.exists() and p.suffix == ".json"
    assert load_transcript(p).model_dump() == t.model_dump()

def test_srt_has_timestamps(make_transcript):
    srt = transcript_to_srt(make_transcript())
    assert "00:00:00,000 --> 00:00:01,500" in srt
    assert "hello revenue" in srt
    assert "Speaker" not in srt

def test_parse_uploaded_json(make_transcript):
    data = make_transcript().model_dump_json().encode()
    assert parse_transcript_file("t.json", data).source_id == "pypl"

def test_malformed_json_raises():
    with pytest.raises(TranscriptFormatError):
        parse_transcript_file("t.json", b"{not json")

def test_unknown_extension_raises():
    with pytest.raises(TranscriptFormatError):
        parse_transcript_file("t.txt", b"hello")
