from types import SimpleNamespace
from src.embeddings import embed_texts, embed_query

class FakeEmbeddings:
    def __init__(self): self.calls = []
    def create(self, model, input):
        batch = list(input); self.calls.append(batch)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(i)]) for i in range(len(batch))])

class FakeClient:
    def __init__(self): self.embeddings = FakeEmbeddings()

def test_batching():
    c = FakeClient()
    out = embed_texts([f"t{i}" for i in range(250)], client=c, model="m", batch_size=100)
    assert len(out) == 250
    assert [len(b) for b in c.embeddings.calls] == [100, 100, 50]

def test_single_query():
    c = FakeClient()
    assert embed_query("hi", client=c, model="m") == [0.0]

def test_empty_input_returns_empty():
    c = FakeClient()
    assert embed_texts([], client=c, model="m") == []
    assert c.embeddings.calls == []
