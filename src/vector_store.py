"""Persistent ChromaDB vector store operations.

A single ChromaDB collection holds every uploaded call's chunks, namespaced by
``source_id``. Chunk ids are ``f"{source_id}::{chunk_index}"``, so re-indexing a
file replaces its previous chunks instead of duplicating them. Callers inject the
client/collection so these helpers stay easy to test against a temporary store.
"""

from __future__ import annotations

from collections.abc import Sequence

import chromadb
from chromadb.config import Settings

from src.models import RetrievedChunk, SourceInfo, TextChunk


class VectorStoreError(Exception):
    """Raised when a ChromaDB operation fails."""


def get_client(persist_dir: str) -> chromadb.ClientAPI:
    """Open a persistent ChromaDB client rooted at ``persist_dir``.

    Telemetry is disabled so the store never phones home and stays quiet in tests.
    """
    try:
        return chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to open vector store at {persist_dir!r}: {exc}"
        ) from exc


def get_or_create_collection(client: chromadb.ClientAPI, name: str) -> chromadb.Collection:
    """Return ``name``, creating it with cosine distance if it does not exist."""
    try:
        return client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to get or create collection {name!r}: {exc}"
        ) from exc


def upsert_chunks(
    collection: chromadb.Collection,
    chunks: Sequence[TextChunk],
    embeddings: Sequence[Sequence[float]],
    indexed_at: float = 0.0,
    source_file: str = "",
) -> None:
    """Insert or replace ``chunks`` (keyed by ``chunk.id``) with their embeddings.

    Metadata scalars stored per chunk: ``source_id``, ``source_file``,
    ``chunk_index``, ``start``, ``end`` and ``indexed_at``.
    """
    chunk_list = list(chunks)
    if not chunk_list:
        return

    embedding_list = list(embeddings)
    if len(embedding_list) != len(chunk_list):
        raise VectorStoreError(
            f"Got {len(embedding_list)} embedding(s) for {len(chunk_list)} chunk(s)"
        )

    try:
        collection.upsert(
            ids=[chunk.id for chunk in chunk_list],
            embeddings=embedding_list,
            documents=[chunk.text for chunk in chunk_list],
            metadatas=[
                {
                    "source_id": chunk.source_id,
                    "source_file": source_file,
                    "chunk_index": chunk.chunk_index,
                    "start": chunk.start,
                    "end": chunk.end,
                    "indexed_at": indexed_at,
                }
                for chunk in chunk_list
            ],
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to upsert {len(chunk_list)} chunk(s): {exc}"
        ) from exc


def query(
    collection: chromadb.Collection,
    query_embedding: Sequence[float],
    top_k: int,
    source_id: str | None = None,
) -> list[RetrievedChunk]:
    """Return the ``top_k`` closest chunks, optionally limited to one ``source_id``.

    ``source_id=None`` searches every source. An empty-string ``source_id`` is a
    real filter (matching no chunks) rather than a search across all sources.
    Chroma's cosine distance is exposed as ``RetrievedChunk.score`` (lower is
    closer).
    """
    where = {"source_id": source_id} if source_id is not None else None
    try:
        result = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            where=where,
        )
    except Exception as exc:
        raise VectorStoreError(f"Failed to query vector store: {exc}") from exc

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    retrieved: list[RetrievedChunk] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        retrieved.append(
            RetrievedChunk(
                text=document,
                start=metadata["start"],
                end=metadata["end"],
                score=distance,
                chunk_index=metadata["chunk_index"],
            )
        )
    return retrieved


def list_sources(collection: chromadb.Collection) -> list[SourceInfo]:
    """Aggregate stored chunks into one ``SourceInfo`` per ``source_id``."""
    try:
        stored = collection.get(include=["metadatas"])
    except Exception as exc:
        raise VectorStoreError(f"Failed to list sources: {exc}") from exc

    aggregates: dict[str, SourceInfo] = {}
    for metadata in stored.get("metadatas") or []:
        source_id = metadata["source_id"]
        existing = aggregates.get(source_id)
        if existing is None:
            aggregates[source_id] = SourceInfo(
                source_id=source_id,
                source_file=metadata.get("source_file", ""),
                chunk_count=1,
                indexed_at=metadata.get("indexed_at", 0.0),
            )
        else:
            existing.chunk_count += 1
            existing.indexed_at = max(
                existing.indexed_at, metadata.get("indexed_at", 0.0)
            )
    return list(aggregates.values())


def latest_source_id(collection: chromadb.Collection) -> str | None:
    """Return the ``source_id`` with the greatest ``indexed_at``, or ``None``."""
    sources = list_sources(collection)
    if not sources:
        return None
    return max(sources, key=lambda source: source.indexed_at).source_id


def delete_source(collection: chromadb.Collection, source_id: str) -> None:
    """Delete every chunk belonging to ``source_id``, leaving other sources intact."""
    try:
        collection.delete(where={"source_id": source_id})
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to delete source {source_id!r}: {exc}"
        ) from exc


def count(collection: chromadb.Collection) -> int:
    """Return the total number of chunks stored across all sources."""
    try:
        return collection.count()
    except Exception as exc:
        raise VectorStoreError(f"Failed to count chunks: {exc}") from exc
