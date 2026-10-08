"""Batched OpenAI embedding helpers.

The OpenAI client is injected by the caller rather than constructed here, so the
functions stay easy to test with a fake client and no API key. Texts are split
into ``batch_size`` slices to respect the embeddings API's per-request limit.
"""

from collections.abc import Sequence


class EmbeddingError(Exception):
    """Raised when the embeddings client fails to produce embeddings."""


def embed_texts(
    texts: Sequence[str],
    client,
    model: str,
    batch_size: int = 100,
) -> list[list[float]]:
    """Embed ``texts`` in batches, returning one vector per input in order.

    ``client`` is an OpenAI-like object exposing
    ``client.embeddings.create(model=..., input=...)``. Empty input returns an
    empty list without contacting the client.
    """
    if batch_size <= 0:
        raise EmbeddingError(f"batch_size must be positive, got {batch_size}")

    vectors: list[list[float]] = []
    items: list[str] = list(texts)
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        try:
            response = client.embeddings.create(model=model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to embed batch of {len(batch)} text(s): {exc}"
            ) from exc
    return vectors


def embed_query(text: str, client, model: str) -> list[float]:
    """Embed a single query string and return its vector."""
    return embed_texts([text], client=client, model=model)[0]
