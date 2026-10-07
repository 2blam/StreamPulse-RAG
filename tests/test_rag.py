from types import SimpleNamespace
import pytest
from src.models import RetrievedChunk, TextChunk
from src.rag import format_timestamp, build_context, build_messages, answer_question
from src.vector_store import get_client, get_or_create_collection, upsert_chunks

def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(63) == "01:03"
    assert format_timestamp(3724) == "1:02:04"

def test_build_context_has_timestamps():
    ctx = build_context([RetrievedChunk(text="revenue up", start=12.0, end=45.0, score=0.1, chunk_index=0)])
    assert "[00:12-00:45]" in ctx and "revenue up" in ctx

def test_build_messages_system_and_context():
    msgs = build_messages("How was revenue?", [RetrievedChunk(text="x", start=0, end=1, score=0, chunk_index=0)])
    assert msgs[0]["role"] == "system"
    assert "MM:SS" in msgs[0]["content"] or "mm:ss" in msgs[0]["content"].lower()
    assert "How was revenue?" in msgs[-1]["content"]

@pytest.fixture
def collection(tmp_path):
    return get_or_create_collection(get_client(str(tmp_path / "c")), "test")

def test_answer_question_stub_llm(collection, make_transcript, tmp_path):
    from src.indexer import index_transcript
    from types import SimpleNamespace
    class FakeEmb:
        def create(self, model, input): return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])
    client = SimpleNamespace(embeddings=FakeEmb())
    index_transcript(make_transcript(), collection, client, settings=None)
    llm = SimpleNamespace(invoke=lambda messages: SimpleNamespace(content="Revenue grew [00:01]"))
    answer, retrieved = answer_question("revenue?", collection, client, settings=SimpleNamespace(embedding_model="m", rag_top_k=6), llm=llm)
    assert "Revenue grew" in answer
    assert retrieved  # citations returned for the UI


def test_default_llm_passes_settings_api_key(collection, make_transcript, monkeypatch):
    """The default ChatOpenAI must receive settings.openai_api_key explicitly."""
    import langchain_openai
    from src.indexer import index_transcript
    from types import SimpleNamespace

    class FakeEmb:
        def create(self, model, input): return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 0.0]) for _ in input])

    client = SimpleNamespace(embeddings=FakeEmb())
    index_transcript(make_transcript(), collection, client, settings=None)

    captured: dict = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def invoke(self, messages):
            return SimpleNamespace(content="ok")

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChat)
    settings = SimpleNamespace(
        embedding_model="m", rag_top_k=6, chat_model="gpt-4o", openai_api_key="sk-test"
    )
    answer_question("revenue?", collection, client, settings=settings, llm=None)

    assert captured["api_key"] == "sk-test"
