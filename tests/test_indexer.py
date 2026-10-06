from src.models import TranscriptSegment
from src.indexer import chunk_transcript, index_transcript
from src.vector_store import get_client, get_or_create_collection, count

def test_windows_preserve_time_span_and_ids(make_transcript):
    segs = [TranscriptSegment(index=i, start=i * 1.0, end=i * 1.0 + 1.0, text="x" * 5) for i in range(20)]
    t = make_transcript(segments=segs)
    chunks = chunk_transcript(t, target_chars=20, overlap_chars=5)
    assert chunks[0].start == 0.0
    assert chunks[-1].end == 20.0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert len({c.id for c in chunks}) == len(chunks)
    assert all(c.text.strip() for c in chunks)

def test_long_segment_is_split_and_time_preserved(make_transcript):
    segs = [TranscriptSegment(index=0, start=3.0, end=9.0, text="y" * 100)]
    chunks = chunk_transcript(make_transcript(segments=segs), target_chars=20, overlap_chars=5)
    assert len(chunks) > 1
    assert all(c.start == 3.0 and c.end == 9.0 for c in chunks)
    assert sum(len(c.text) for c in chunks) >= 100  # no text lost

def test_overlap_start_inherited_from_trailing_segment(make_transcript):
    segs = [TranscriptSegment(index=i, start=i * 10.0, end=i * 10.0 + 10.0, text=chr(65 + i) * 10) for i in range(6)]
    t = make_transcript(segments=segs)
    chunks = chunk_transcript(t, target_chars=20, overlap_chars=5)
    assert chunks[0].start == 0.0 and chunks[0].end == 20.0
    assert chunks[1].start == 10.0  # overlap is the tail of segment 1
    assert all(c.id == f"{c.source_id}::{c.chunk_index}" for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)

def test_index_transcript_uses_embeddings(make_transcript, tmp_path):
    from types import SimpleNamespace
    class FakeEmb:
        def create(self, model, input): return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])
    client = SimpleNamespace(embeddings=FakeEmb())
    coll = get_or_create_collection(get_client(str(tmp_path / "c")), "test")
    n = index_transcript(make_transcript(), coll, client, settings=None)
    assert n == count(coll) and n > 0
