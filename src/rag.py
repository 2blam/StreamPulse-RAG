"""Retrieval-augmented question answering over indexed earnings-call chunks.

A user question is embedded with the same OpenAI embedding model used at index
time, the closest transcript chunks are pulled from ChromaDB, and GPT-4o is
asked to answer using only that retrieved context, citing ``[MM:SS]`` spans.

The embedding client and LLM are injected so the module stays testable without
an API key; the default ``ChatOpenAI`` is built only when ``llm is None``.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.embeddings import embed_query
from src.models import RetrievedChunk
from src.vector_store import query as vector_store_query

SYSTEM_PROMPT = (
    "You are a financial analyst assistant for FinTech earnings calls. "
    "Answer the user's question using ONLY the provided transcript context. "
    "Cite the supporting timestamp(s) inline as [MM:SS] (use [H:MM:SS] for "
    "calls over an hour), matching the timestamps shown in the context. "
    "If the context does not contain enough information to answer, say so "
    "clearly rather than guessing or inventing figures."
)

NO_CONTEXT_MESSAGE = (
    "I couldn't find any relevant context in the indexed earnings calls to "
    "answer that question."
)


def format_timestamp(seconds: float) -> str:
    """Format ``seconds`` as ``MM:SS``, or ``H:MM:SS`` at/over one hour."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_context(retrieved: Sequence[RetrievedChunk]) -> str:
    """Render retrieved chunks as one ``[MM:SS-MM:SS] text`` line each."""
    return "\n".join(
        f"[{format_timestamp(c.start)}-{format_timestamp(c.end)}] {c.text}"
        for c in retrieved
    )


def build_messages(
    question: str, retrieved: Sequence[RetrievedChunk]
) -> list[dict]:
    """Build the chat messages: system prompt first, then context + question."""
    context = build_context(retrieved)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def answer_question(
    question: str,
    collection,
    client,
    settings,
    source_id: str | None = None,
    top_k: int | None = None,
    llm=None,
) -> tuple[str, list[RetrievedChunk]]:
    """Answer ``question`` from the top retrieved chunks, citing timestamps.

    Returns ``(answer_text, retrieved_chunks)`` so the UI can show the sources
    behind the answer. When nothing is retrieved the LLM is not called and a
    graceful "no relevant context" message is returned instead.
    """
    if top_k is None:
        top_k = getattr(settings, "rag_top_k", 6)

    query_embedding = embed_query(
        question,
        client,
        getattr(settings, "embedding_model", "text-embedding-3-small"),
    )
    retrieved = vector_store_query(
        collection, query_embedding, top_k, source_id=source_id
    )
    if not retrieved:
        return NO_CONTEXT_MESSAGE, []

    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.chat_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )

    response = llm.invoke(build_messages(question, retrieved))
    return response.content, retrieved
