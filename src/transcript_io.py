"""Persist and re-load transcript artifacts as JSON and SRT.

Transcripts can be downloaded from the UI as JSON (lossless, re-uploadable) or
SRT (human-readable subtitles). ``parse_transcript_file`` accepts either format
so a previously downloaded transcript can be uploaded to skip re-transcription.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re

from src.models import Transcript, TranscriptSegment


class TranscriptFormatError(Exception):
    """Raised when an uploaded transcript cannot be parsed as JSON or SRT."""


_SLUG_RE = re.compile(r"[^a-z0-9]+")
_TIMESTAMP_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")


def slugify(name: str) -> str:
    """Lowercase ``name`` (minus any file extension) collapsing non-alphanumerics.

    Each run of non-alphanumeric characters becomes a single ``_`` and any
    leading/trailing ``_`` is stripped, e.g. ``"PYPL_Q4_2025_Earnings_Call.mp3"``
    becomes ``"pypl_q4_2025_earnings_call"``.

    A name that slugs to the empty string (e.g. ``"!!!.mp3"``) falls back to a
    deterministic ``"source_<sha1>"`` id so a source is never unnamed — an empty
    ``source_id`` would otherwise disable source-scoped filtering.

    Collisions are accepted by design: two differently-named files can normalize
    to the same slug (e.g. ``"Q4 2025.mp3"`` and ``"q4-2025.wav"``) and are
    treated as the duplicate-overwrite case rather than distinct sources.
    """
    stem = Path(name).stem
    slug = _SLUG_RE.sub("_", stem.lower()).strip("_")
    if slug:
        return slug
    return "source_" + hashlib.sha1(name.encode()).hexdigest()[:8]


def _format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _parse_timestamp(value: str) -> float:
    match = _TIMESTAMP_RE.search(value)
    if not match:
        raise TranscriptFormatError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, secs, ms = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(ms.ljust(3, "0")) / 1000.0


def save_transcript(t: Transcript, out_dir="transcripts") -> Path:
    """Write ``t`` to ``<out_dir>/<source_id>.json`` and return the path."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / f"{t.source_id}.json"
    dest.write_text(t.model_dump_json(indent=2), encoding="utf-8")
    return dest


def load_transcript(path) -> Transcript:
    """Load a transcript previously written by :func:`save_transcript`."""
    return Transcript.model_validate_json(Path(path).read_text(encoding="utf-8"))


def transcript_to_srt(t: Transcript) -> str:
    """Render ``t`` as SRT with ``HH:MM:SS,mmm --> HH:MM:SS,mmm`` cues."""
    blocks: list[str] = []
    for number, seg in enumerate(t.segments, start=1):
        blocks.append(
            f"{number}\n"
            f"{_format_timestamp(seg.start)} --> {_format_timestamp(seg.end)}\n"
            f"{seg.text}\n"
        )
    return "\n".join(blocks)


def _parse_srt(text: str, filename: str) -> Transcript:
    segments: list[TranscriptSegment] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        # Optional leading numeric cue index; locate the timestamp line instead.
        ts_idx = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if ts_idx is None:
            continue
        start_raw, end_raw = lines[ts_idx].split("-->", 1)
        start = _parse_timestamp(start_raw)
        end = _parse_timestamp(end_raw)
        cue_text = " ".join(line.strip() for line in lines[ts_idx + 1:]).strip()
        segments.append(
            TranscriptSegment(index=len(segments), start=start, end=end, text=cue_text)
        )

    if not segments:
        raise TranscriptFormatError(f"No SRT cues found in {filename!r}")

    return Transcript(
        source_id=slugify(filename),
        source_file=filename,
        duration_seconds=segments[-1].end,
        transcription_model="uploaded",
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        segments=segments,
    )


def parse_transcript_file(filename: str, data: bytes) -> Transcript:
    """Parse an uploaded ``.json`` or ``.srt`` transcript.

    Raises :class:`TranscriptFormatError` for malformed input or any other
    extension, so callers never see a raw traceback.
    """
    suffix = Path(filename).suffix.lower()
    try:
        text = data.decode("utf-8")
    except (UnicodeDecodeError, AttributeError) as exc:
        raise TranscriptFormatError(f"Could not decode {filename!r}: {exc}") from exc

    if suffix == ".json":
        try:
            return Transcript.model_validate_json(text)
        except Exception as exc:
            raise TranscriptFormatError(f"Invalid transcript JSON in {filename!r}: {exc}") from exc

    if suffix == ".srt":
        return _parse_srt(text, filename)

    raise TranscriptFormatError(f"Unsupported transcript extension: {filename!r}")
