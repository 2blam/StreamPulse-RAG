import pytest
from src.models import TextChunk
from src.vector_store import (get_client, get_or_create_collection, upsert_chunks,
                              query, list_sources, latest_source_id, delete_source, count)

@pytest.fixture
def collection(tmp_path):
    return get_or_create_collection(get_client(str(tmp_path / "chroma")), "test")

def _chunk(cid, sid, i, text, start=0.0, end=5.0):
    return TextChunk(id=cid, source_id=sid, chunk_index=i, start=start, end=end, text=text)

def test_upsert_query_roundtrip(collection):
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "revenue growth")], [[1.0, 0.0, 0.0]], indexed_at=100.0)
    res = query(collection, [1.0, 0.0, 0.0], top_k=1)
    assert res[0].text == "revenue growth" and res[0].start == 0.0

def test_source_filter(collection):
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "alpha"), _chunk("b::0", "b", 0, "beta")],
                  [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], indexed_at=100.0)
    res = query(collection, [1.0, 0.0, 0.0], top_k=10, source_id="b")
    assert {r.chunk_index for r in res} and all("beta" == r.text for r in res)

def test_list_sources_and_latest(collection):
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "x")], [[1.0, 0.0, 0.0]], indexed_at=100.0)
    upsert_chunks(collection, [_chunk("b::0", "b", 0, "y")], [[1.0, 0.0, 0.0]], indexed_at=200.0)
    infos = {s.source_id: s for s in list_sources(collection)}
    assert infos["a"].chunk_count == 1
    assert latest_source_id(collection) == "b"

def test_duplicate_filename_overwrites(collection):
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "old")], [[1.0, 0.0, 0.0]], indexed_at=1.0)
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "new")], [[1.0, 0.0, 0.0]], indexed_at=2.0)
    assert count(collection) == 1
    assert query(collection, [1.0, 0.0, 0.0], top_k=1)[0].text == "new"

def test_delete_source(collection):
    upsert_chunks(collection, [_chunk("a::0", "a", 0, "x"), _chunk("b::0", "b", 0, "y")],
                  [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], indexed_at=1.0)
    delete_source(collection, "a")
    assert {s.source_id for s in list_sources(collection)} == {"b"}
