from pathlib import Path
import pytest
from pydub import AudioSegment
from src.audio import split_audio, probe_duration, cleanup_chunks, AudioError

@pytest.fixture
def tone_mp3(tmp_path):
    p = tmp_path / "tone.mp3"
    # 12 s of a quiet tone (silence is fine for chunking tests)
    AudioSegment.silent(duration=12_000).export(p, format="mp3")
    return p

def test_probe_duration(tone_mp3):
    assert abs(probe_duration(tone_mp3) - 12.0) < 0.5

def test_split_offsets_and_files(tone_mp3, tmp_path):
    chunks = split_audio(tone_mp3, chunk_seconds=5, out_dir=tmp_path / "out")
    assert [round(c.start, 2) for c in chunks] == [0.0, 5.0, 10.0]
    assert [round(c.end, 2) for c in chunks] == [5.0, 10.0, 12.0]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert all(Path(c.path).exists() for c in chunks)
    cleanup_chunks(chunks)
    assert not Path(chunks[0].path).exists()

def test_limit_seconds(tone_mp3, tmp_path):
    chunks = split_audio(tone_mp3, chunk_seconds=5, limit_seconds=7, out_dir=tmp_path / "out")
    assert len(chunks) == 2
    assert round(chunks[-1].end, 2) == 7.0

def test_missing_file_raises(tmp_path):
    with pytest.raises(AudioError):
        split_audio(tmp_path / "nope.mp3", out_dir=tmp_path)

def test_short_audio_single_chunk(tmp_path):
    p = tmp_path / "tiny.mp3"
    AudioSegment.silent(duration=800).export(p, format="mp3")  # 0.8 s
    chunks = split_audio(p, chunk_seconds=300, out_dir=tmp_path / "out")
    assert len(chunks) == 1
    assert round(chunks[0].end, 1) <= 1.0
