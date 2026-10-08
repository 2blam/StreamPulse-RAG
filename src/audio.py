"""Split long earnings-call audio into timestamped, transcriber-friendly chunks.

Chunks are exported as mono 16 kHz 64 kbps MP3 files. ``AudioChunk.start`` and
``AudioChunk.end`` are *global* offsets in seconds into the original recording,
so timestamps stay meaningful after transcription.
"""

from dataclasses import dataclass
from pathlib import Path
import tempfile

from pydub import AudioSegment


class AudioError(Exception):
    """Raised when an audio file cannot be decoded or a chunk cannot be exported."""


@dataclass
class AudioChunk:
    index: int
    start: float          # seconds, global offset into the original audio
    end: float            # seconds, global offset into the original audio
    path: Path            # exported mono 16 kHz mp3 chunk


def _load(path) -> AudioSegment:
    try:
        return AudioSegment.from_file(path)
    except Exception as exc:  # missing file, unsupported codec, corrupt data...
        raise AudioError(f"Failed to decode audio file {path}: {exc}") from exc


def probe_duration(path) -> float:
    """Return the duration of ``path`` in seconds."""
    return len(_load(path)) / 1000.0


def split_audio(
    path,
    chunk_seconds: int = 300,
    limit_seconds: float | None = None,
    out_dir=None,
    sample_rate: int = 16000,
    bitrate: str = "64k",
) -> list[AudioChunk]:
    """Split ``path`` into ``chunk_seconds``-long MP3 chunks.

    ``limit_seconds`` truncates the audio to its first N seconds before chunking
    (useful for quick excerpt tests). Each chunk is exported as mono at
    ``sample_rate`` and ``bitrate``. When ``out_dir`` is omitted a fresh
    temporary directory is created.
    """
    audio = _load(path)

    total_ms = len(audio)
    if limit_seconds is not None:
        total_ms = min(total_ms, int(round(limit_seconds * 1000)))

    if total_ms <= 0:
        return []

    chunk_ms = int(chunk_seconds * 1000)
    if chunk_ms <= 0:
        raise AudioError(f"chunk_seconds must be positive, got {chunk_seconds}")

    out_path = Path(out_dir) if out_dir is not None else Path(tempfile.mkdtemp())
    out_path.mkdir(parents=True, exist_ok=True)

    chunks: list[AudioChunk] = []
    for index, start_ms in enumerate(range(0, total_ms, chunk_ms)):
        end_ms = min(start_ms + chunk_ms, total_ms)
        segment = audio[start_ms:end_ms]
        dest = out_path / f"chunk_{index:04d}.mp3"
        try:
            segment.export(
                dest,
                format="mp3",
                bitrate=bitrate,
                parameters=["-ac", "1", "-ar", str(sample_rate)],
            )
        except Exception as exc:
            raise AudioError(f"Failed to export chunk {index} for {path}: {exc}") from exc
        chunks.append(
            AudioChunk(
                index=index,
                start=start_ms / 1000.0,
                end=end_ms / 1000.0,
                path=dest,
            )
        )

    return chunks


def cleanup_chunks(chunks) -> None:
    """Delete the exported chunk files, ignoring any that are already gone."""
    for chunk in chunks:
        chunk_path = Path(chunk.path)
        if chunk_path.exists():
            chunk_path.unlink()
