#!/usr/bin/env python3
"""StreamPulse-RAG development smoke test.

Validates that the local development environment is correctly set up:

  1. Python version is 3.12+ (project venv currently runs 3.13)
  2. All required third-party packages import cleanly
  3. ``ffmpeg`` is available on PATH (required by pydub)
  4. Environment configuration (.env / OPENAI_API_KEY) is present
  5. The ChromaDB vector store can be created and queried locally

Exit code is 0 when all required checks pass, 1 otherwise.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Tiny check harness
# ---------------------------------------------------------------------------
PASSED: list[str] = []
FAILED: list[str] = []
WARNED: list[str] = []


def check(label: str, fn) -> None:
    """Run a check callable; store result and print status."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - smoke test reports everything
        FAILED.append(f"{label}: {exc}")
        print(f"  [FAIL] {label} -> {exc}")
    else:
        PASSED.append(label)
        print(f"  [ OK ] {label}")


def warn(label: str, message: str) -> None:
    WARNED.append(f"{label}: {message}")
    print(f"  [WARN] {label} -> {message}")


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_python_version() -> None:
    major, minor = sys.version_info[:2]
    assert major == 3 and minor >= 12, (
        f"expected Python 3.12+, running {major}.{minor} ({sys.executable})"
    )


def check_imports() -> None:
    modules = {
        "streamlit": "streamlit",
        "openai": "openai",
        "chromadb": "chromadb",
        "pydantic": "pydantic",
        "python-dotenv": "dotenv",
        "plotly": "plotly",
        "langchain-openai": "langchain_openai",
        "langchain-community": "langchain_community",
        "pydub": "pydub",
    }
    for display, module in modules.items():
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise AssertionError(f"cannot import {display!r}: {exc}") from exc


def check_ffmpeg() -> None:
    from pydub.utils import which

    path = which("ffmpeg")
    assert path, "ffmpeg not found on PATH (required by pydub for MP3 decoding)"


def check_env_config() -> None:
    from dotenv import load_dotenv

    env_file = ROOT / ".env"
    if env_file.exists():
        load_dotenv(env_file)

    key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL_NAME", "")
    if not key:
        warn(
            "environment config",
            "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key "
            "before running the RAG engine.",
        )
    if model and model != "gpt-4o":
        warn("environment config", f"OPENAI_MODEL_NAME is {model!r} (default: gpt-4o)")


def check_chromadb() -> None:
    import chromadb

    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="smoke_test")
    collection.add(
        ids=["doc-1"],
        documents=["PayPal reported cross-border volume growth."],
        # Precomputed embeddings avoid downloading an embedding model in the test.
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        metadatas=[{"start": 0.0, "end": 4.2}],
    )
    result = collection.query(query_embeddings=[[0.1, 0.2, 0.3, 0.4]], n_results=1)
    assert result["ids"][0][0] == "doc-1", "ChromaDB query returned unexpected result"
    client.delete_collection(name="smoke_test")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main() -> int:
    print("StreamPulse-RAG — development smoke test")
    print(f"Python : {sys.version.split()[0]}  ({sys.executable})")
    print(f"Project: {ROOT}")
    print("-" * 60)

    check("Python 3.12+", check_python_version)
    check("core package imports", check_imports)
    check("ffmpeg on PATH", check_ffmpeg)
    check("environment config", check_env_config)
    check("ChromaDB vector store", check_chromadb)

    print("-" * 60)
    print(f"Passed: {len(PASSED)}  Failed: {len(FAILED)}  Warnings: {len(WARNED)}")

    if FAILED:
        print("\nFAILURES:")
        for item in FAILED:
            print(f"  - {item}")
        print("\nResult: SMOKE TEST FAILED")
        return 1

    if WARNED:
        print("\nResult: SMOKE TEST PASSED (with warnings)")
        return 0

    print("\nResult: SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
