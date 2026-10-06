"""StreamPulse-RAG — Streamlit UI for FinTech earnings-call intelligence.

This is the user-facing surface: it configures the app in the sidebar, ingests
audio (transcription) or pre-made transcripts into the ChromaDB vector store,
and answers analyst questions with timestamped citations back to the call.

Every network-touching action is guarded behind ``settings.has_api_key`` so the
app boots cleanly (and shows guidance) when no OpenAI key is configured.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from langchain_openai import ChatOpenAI

from src.config import get_openai_client, load_settings
from src.pipeline import ingest_audio, ingest_transcript
from src.rag import answer_question, format_timestamp
from src.transcript_io import TranscriptFormatError, parse_transcript_file, transcript_to_srt
from src.vector_store import (
    delete_source,
    get_client,
    get_or_create_collection,
    latest_source_id,
    list_sources,
)

st.set_page_config(page_title="StreamPulse-RAG", layout="wide")

CHAT_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"]
PREVIEW_SEGMENTS = 20
MISSING_KEY_MESSAGE = "OPENAI_API_KEY is not set. Add it to .env and restart."


@st.cache_resource
def _load_collection(chroma_dir: str, collection_name: str):
    """Open (and cache) the persistent ChromaDB collection."""
    return get_or_create_collection(get_client(chroma_dir), collection_name)


def _progress_bar(bar, label: str):
    """Adapt pipeline ``(current, total)`` callbacks onto a Streamlit progress bar."""

    def update(current: int, total: int) -> None:
        fraction = (current / total) if total else 0.0
        bar.progress(
            min(max(fraction, 0.0), 1.0),
            text=f"{label} {current}/{total}",
        )

    return update


def _render_transcript_preview(transcript, key_prefix: str) -> None:
    """Show the first segments plus JSON/SRT download buttons for a transcript."""
    st.markdown(f"**Preview** — {transcript.source_file} · {len(transcript.segments)} segments")
    for segment in transcript.segments[:PREVIEW_SEGMENTS]:
        st.write(f"[{format_timestamp(segment.start)}] {segment.text}")
    if len(transcript.segments) > PREVIEW_SEGMENTS:
        st.caption(f"… {len(transcript.segments) - PREVIEW_SEGMENTS} more segments")

    col_json, col_srt = st.columns(2)
    col_json.download_button(
        "Download JSON",
        data=transcript.model_dump_json(indent=2),
        file_name=f"{transcript.source_id}.json",
        mime="application/json",
        key=f"{key_prefix}_json",
    )
    col_srt.download_button(
        "Download SRT",
        data=transcript_to_srt(transcript),
        file_name=f"{transcript.source_id}.srt",
        mime="text/plain",
        key=f"{key_prefix}_srt",
    )


settings = load_settings()
collection = _load_collection(settings.chroma_dir, settings.collection_name)
has_key = settings.has_api_key

# --- Sidebar -----------------------------------------------------------------

st.sidebar.title("StreamPulse-RAG")

if has_key:
    st.sidebar.success("OPENAI_API_KEY detected.")
else:
    st.sidebar.error(MISSING_KEY_MESSAGE)

chat_model = st.sidebar.selectbox(
    "Chat model",
    CHAT_MODELS,
    index=CHAT_MODELS.index(settings.chat_model)
    if settings.chat_model in CHAT_MODELS
    else 0,
)

sources = list_sources(collection)
labels = [f"{s.source_file} ({s.source_id})" for s in sources]
latest = latest_source_id(collection)
default_index = next(
    (i for i, s in enumerate(sources) if s.source_id == latest),
    0,
)

selected_label = st.sidebar.selectbox(
    "Source",
    labels,
    index=default_index,
    disabled=not labels,
)
selected_source_id = (
    sources[labels.index(selected_label)].source_id if labels else None
)

scope = st.sidebar.radio("Search scope", ["Selected call", "All calls"], index=0)

st.sidebar.subheader("Indexed sources")
if not sources:
    st.sidebar.caption("No calls indexed yet.")
for source in sources:
    info_col, delete_col = st.sidebar.columns([4, 1])
    info_col.markdown(
        f"**{source.source_file}**  \n`{source.source_id}` · {source.chunk_count} chunks"
    )
    if delete_col.button("🗑", key=f"delete_{source.source_id}", help="Delete source"):
        delete_source(collection, source.source_id)
        st.rerun()

# --- Ingestion tabs ----------------------------------------------------------

tab_audio, tab_transcript = st.tabs(["Transcribe Audio", "Upload Transcript"])

with tab_audio:
    st.subheader("Transcribe an earnings call")
    audio_file = st.file_uploader("Upload audio", type=["mp3", "wav"], key="audio_uploader")

    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        st.audio(audio_bytes)

        if st.button("Transcribe", disabled=not has_key, key="transcribe_button"):
            if not has_key:
                st.warning("OPENAI_API_KEY is required to transcribe audio.")
            else:
                suffix = Path(audio_file.name).suffix or ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                bar = st.progress(0.0, text="Starting…")
                try:
                    with st.spinner("Transcribing and indexing…"):
                        result = ingest_audio(
                            tmp_path,
                            collection,
                            get_openai_client(settings.openai_api_key),
                            settings,
                            progress=_progress_bar(bar, "Chunks"),
                        )
                    st.session_state["transcript"] = result.transcript
                    st.success(
                        f"Transcribed {result.transcript.source_file} and indexed "
                        f"{result.chunk_count} chunks."
                    )
                except Exception as exc:  # surface failures in the UI, not a traceback
                    st.error(f"Transcription failed: {exc}")
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
                    bar.empty()

    transcript = st.session_state.get("transcript")
    if transcript is not None:
        st.divider()
        _render_transcript_preview(transcript, "audio")
        if st.button("Index", disabled=not has_key, key="index_transcribed"):
            try:
                with st.spinner("Indexing…"):
                    n = ingest_transcript(
                        transcript,
                        collection,
                        get_openai_client(settings.openai_api_key),
                        settings,
                    )
                st.success(f"Indexed {n} chunks from {transcript.source_file}.")
            except Exception as exc:  # surface failures in the UI, not a traceback
                st.error(f"Indexing failed: {exc}")

with tab_transcript:
    st.subheader("Upload a transcript")
    st.caption("Re-upload a previously downloaded JSON or SRT transcript to skip transcription.")
    transcript_file = st.file_uploader(
        "Upload transcript", type=["json", "srt"], key="transcript_uploader"
    )

    if transcript_file is not None:
        try:
            parsed = parse_transcript_file(transcript_file.name, transcript_file.getvalue())
        except TranscriptFormatError as exc:
            st.error(str(exc))
        else:
            st.session_state["uploaded_transcript"] = parsed
            _render_transcript_preview(parsed, "upload")
            if st.button("Index", disabled=not has_key, key="index_uploaded"):
                try:
                    with st.spinner("Indexing…"):
                        n = ingest_transcript(
                            parsed,
                            collection,
                            get_openai_client(settings.openai_api_key),
                            settings,
                        )
                    st.success(f"Indexed {n} chunks from {parsed.source_file}.")
                except Exception as exc:  # surface failures in the UI, not a traceback
                    st.error(f"Indexing failed: {exc}")

# --- Chat --------------------------------------------------------------------

st.divider()
st.subheader("Ask about the calls")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("chunks"):
            with st.expander("Retrieved chunks"):
                for line in message["chunks"]:
                    st.markdown(line)

question = st.chat_input("Ask a question about the earnings calls")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not has_key:
            answer = MISSING_KEY_MESSAGE
            st.warning(answer)
            chunks: list[str] = []
        elif not sources:
            answer = "No calls are indexed yet. Upload audio or a transcript first."
            st.info(answer)
            chunks = []
        else:
            source_id = None if scope == "All calls" else selected_source_id
            try:
                with st.spinner("Thinking…"):
                    answer, retrieved = answer_question(
                        question,
                        collection,
                        get_openai_client(settings.openai_api_key),
                        settings,
                        source_id=source_id,
                        llm=ChatOpenAI(
                            model=chat_model,
                            temperature=0,
                            api_key=settings.openai_api_key,
                        ),
                    )
                chunks = [
                    f"[{format_timestamp(c.start)}-{format_timestamp(c.end)}] {c.text}"
                    for c in retrieved
                ]
                st.markdown(answer)
                if chunks:
                    with st.expander("Retrieved chunks"):
                        for line in chunks:
                            st.markdown(line)
            except Exception as exc:  # surface failures in the UI, not a traceback
                answer = f"Answering failed: {exc}"
                chunks = []
                st.error(answer)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "chunks": chunks}
    )
