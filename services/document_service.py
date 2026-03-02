"""
app/services/document_service.py
──────────────────────────────────
Orchestrates the PDF ingest pipeline and database lifecycle.

Services sit between API routes and the RAG/storage layer.
Routes stay thin; RAG modules stay pure.
"""

from __future__ import annotations

import gc
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from core.config import get_settings
from core.exceptions import FileTooLargeError, IngestError, UnsupportedFileTypeError
from rag.loader import load_and_chunk
from rag import vectorstore as vs
from schemas.document import IngestResponse, StatusResponse

_ALLOWED_EXTENSIONS = {".pdf"}


# ─────────────────────────────────────────────────────────────────────────────
# Public service functions
# ─────────────────────────────────────────────────────────────────────────────

async def ingest_document(file: UploadFile) -> IngestResponse:
    """
    Validate, save, chunk and embed a PDF file.

    Steps
    -----
    1. Validate extension and file size
    2. Persist the raw file to the uploads directory
    3. Load and chunk via rag.loader
    4. Embed and store via rag.vectorstore
    5. Return structured IngestResponse
    """
    cfg = get_settings()

    # ── 1. Validate extension ──────────────────────────
    ext = Path(file.filename).suffix.lower()
    if ext not in _ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(file.filename)

    # ── 2. Validate size (read into memory once) ───────
    content = await file.read()
    if len(content) > cfg.max_upload_bytes:
        raise FileTooLargeError(cfg.max_upload_mb)

    # ── 3. Save to disk ────────────────────────────────
    save_path = cfg.upload_path / file.filename
    save_path.write_bytes(content)

    # ── 4. Chunk + embed ───────────────────────────────
    t0 = time.perf_counter()
    try:
        chunks, stats = load_and_chunk(str(save_path))
        vs.add_documents(chunks)
    except Exception as exc:
        raise IngestError(str(exc)) from exc

    elapsed = round(time.perf_counter() - t0, 2)

    return IngestResponse(
        filename=file.filename,
        pages=stats["pages"],
        chunks=stats["chunks"],
        elapsed=elapsed,
        total_chunks=vs.get_chunk_count(),
        ingested_files=vs.get_ingested_filenames(),
    )


def get_status() -> StatusResponse:
    """Return a snapshot of the vector store health."""
    cfg = get_settings()
    return StatusResponse(
        db_ready=vs.db_is_ready(),
        total_chunks=vs.get_chunk_count(),
        ingested_files=vs.get_ingested_filenames(),
        groq_key_configured=bool(cfg.groq_api_key),
    )


def reset_database() -> None:
    """
    Wipe ChromaDB and all uploaded files.

    Sequence
    --------
    1. release_store()           — stop _system, close all file handles
    2. _rmtree_windows_safe()    — delete chroma_db/ with retry + fallback
    3. delete uploads/           — wipe raw PDFs
    4. recreate uploads/         — empty dir ready for next upload

    After this call the next vs.add_documents() / vs.get_store() will
    create a fresh PersistentClient + chroma.sqlite3 from scratch.
    The tenant bootstrap runs cleanly because the directory is gone.
    """
    cfg = get_settings()

    # ── 1. Stop ChromaDB — closes hnswlib mmap + SQLite ─
    vs.release_store()

    # ── 2. Delete chroma_db/ ───────────────────────────
    _rmtree_windows_safe(cfg.chroma_path)

    # ── 3 & 4. Wipe + recreate uploads/ ────────────────
    _rmtree_windows_safe(cfg.upload_path)
    cfg.upload_path.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _force_remove_readonly(func, path, _exc_info):
    """
    onerror / onexc handler for shutil.rmtree.

    Windows marks SQLite WAL and SHM files read-only.
    Strip the flag and retry the removal.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass


def _rmtree_windows_safe(
    path: Path,
    retries: int = 8,
    base_delay: float = 0.25,
) -> None:
    """
    Delete a directory tree robustly on Windows.

    Strategy
    --------
    1. shutil.rmtree with a read-only error handler.
    2. On PermissionError: gc.collect() + exponential sleep, then retry.
    3. Final fallback: per-file os.unlink (uses DeleteFileW, a different
       Win32 call path that can succeed when rmtree's path fails).

    Parameters
    ----------
    path       : directory to delete; skipped if it doesn't exist
    retries    : max attempts (default 8, ~7 s total)
    base_delay : initial sleep in seconds; doubles each retry (max 2 s)
    """
    if not path.exists():
        return

    last_exc: Optional[Exception] = None
    delay = base_delay

    for attempt in range(1, retries + 1):
        try:
            try:
                shutil.rmtree(path, onexc=_force_remove_readonly)   # Python 3.12+
            except TypeError:
                shutil.rmtree(path, onerror=_force_remove_readonly)  # Python < 3.12
            return  # ── success ──

        except PermissionError as exc:
            last_exc = exc
            if attempt == retries:
                break
            gc.collect()
            time.sleep(delay)
            delay = min(delay * 2, 2.0)

    # ── Last resort: unlink files individually ────────────────────────────────
    try:
        _unlink_tree(path)
        return
    except Exception as fallback_exc:
        raise PermissionError(
            f"Could not delete '{path}' after {retries} attempts.\n"
            f"Last rmtree error   : {last_exc}\n"
            f"Last per-file error : {fallback_exc}\n\n"
            "Workaround: stop the server, delete chroma_db/ manually, restart."
        ) from last_exc


def _unlink_tree(path: Path) -> None:
    """
    Recursively unlink files then rmdir directories (deepest first).

    Uses os.unlink (DeleteFileW) instead of rmtree's internal path — a
    different Win32 code path that can succeed on files rmtree cannot delete.
    """
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            os.chmod(child, stat.S_IWRITE)
        except Exception:
            pass
        try:
            if child.is_file() or child.is_symlink():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        except Exception:
            pass

    try:
        path.rmdir()
    except Exception as exc:
        raise PermissionError(f"Could not remove root dir '{path}': {exc}") from exc