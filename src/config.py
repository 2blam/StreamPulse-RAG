from dataclasses import dataclass
import os

from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = ""
    chat_model: str = "gpt-4o"
    transcription_model: str = "whisper-1"
    embedding_model: str = "text-embedding-3-small"
    chroma_dir: str = "chroma_db"
    collection_name: str = "earnings_calls"
    audio_chunk_seconds: int = 300
    embed_batch_size: int = 100
    rag_top_k: int = 6
    chunk_target_chars: int = 1200
    chunk_overlap_chars: int = 200

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key.strip())


def _int_env(name: str, default: int) -> int:
    """Read ``name`` as an int, falling back to ``default`` when unset/invalid."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_settings(env_file: str | None = ".env") -> Settings:
    if env_file:
        load_dotenv(env_file)  # does not override existing env vars
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chat_model=os.getenv("OPENAI_MODEL_NAME", "gpt-4o"),
        transcription_model=os.getenv("TRANSCRIPTION_MODEL", "whisper-1"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chroma_dir=os.getenv("CHROMA_DIR", "chroma_db"),
        audio_chunk_seconds=_int_env("AUDIO_CHUNK_SECONDS", 300),
    )


def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)
