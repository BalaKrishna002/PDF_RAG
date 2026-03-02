"""
app/rag/vectorstore.py
──────────────────────
ChromaDB vector store — lifecycle management.

Architecture
------------
We own the chromadb.PersistentClient explicitly (rather than letting the
LangChain Chroma wrapper create an implicit one) so we have full control
over the client lifecycle — critical for Windows file-handle management.

One PersistentClient is shared for the process lifetime.  It is replaced
only after a full reset (release_store + directory deletion).

Error: "Could not connect to tenant default_tenant"
----------------------------------------------------
This error occurs when:
  a) A Chroma object is opened against a directory that was deleted after
     its internal _system was stopped — the SQLite db no longer exists so
     the tenant bootstrap fails.
  b) The LangChain Chroma wrapper creates a NEW implicit PersistentClient
     against a half-deleted or freshly empty chroma_db/ directory that
     hasn't been fully initialised yet (no chroma.sqlite3 present).

Fix: we control client creation ourselves via _get_or_create_client().
  - On first call (or after reset) we create a fresh PersistentClient,
    which runs the full tenant/database bootstrap and writes chroma.sqlite3.
  - We never reuse a client whose _system has been stopped.
  - We never open a Chroma LangChain object against a directory that was
    previously managed by a now-stopped client.

Windows WinError 32
-------------------
The hnswlib C-extension keeps data_level0.bin exclusively locked.
Only _client._system.stop() releases it (not reset(), not GC).
See release_store() for the full shutdown sequence.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Optional

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.config import get_settings
from rag.embeddings import get_embedding_model

# ── Module-level singletons ───────────────────────────────────────────────────
_client: Optional[chromadb.PersistentClient] = None
_store: Optional[Chroma] = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal: client + store factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_or_create_client() -> chromadb.PersistentClient:
    """
    Return the cached PersistentClient, creating one if necessary.

    Creating the client (not just the Chroma wrapper) ensures ChromaDB runs
    its full bootstrap: creates chroma.sqlite3, registers the default tenant
    and database — so the "Could not connect to tenant default_tenant" error
    can never occur on a fresh or post-reset directory.
    """
    global _client
    cfg = get_settings()

    if _client is None:
        # Ensure the directory exists before handing it to ChromaDB
        cfg.chroma_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(cfg.chroma_path))

    return _client


def _build_store(client: chromadb.PersistentClient) -> Chroma:
    """Wrap an existing PersistentClient in the LangChain Chroma interface."""
    cfg = get_settings()
    return Chroma(
        client=client,
        collection_name=cfg.chroma_collection,
        embedding_function=get_embedding_model(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public lifecycle helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_store() -> Optional[Chroma]:
    """
    Return the LangChain Chroma wrapper, or None if no docs have been ingested.

    Lazily creates the PersistentClient (and chroma.sqlite3) on first call.
    Returns None only when the collection is genuinely empty so callers can
    distinguish "ready" from "needs first upload".
    """
    global _store

    if _store is None:
        client = _get_or_create_client()
        cfg    = get_settings()

        # Check whether the collection already has data
        try:
            col   = client.get_or_create_collection(cfg.chroma_collection)
            count = col.count()
        except Exception:
            count = 0

        if count > 0:
            _store = _build_store(client)

    return _store


def invalidate_store() -> None:
    """
    Drop the LangChain wrapper reference so it is rebuilt on next access.

    Does NOT close the underlying PersistentClient — the client stays alive
    and its file handles remain open (correct behaviour for normal operation).
    Call this after add_documents() so metadata helpers pick up new counts.
    """
    global _store
    _store = None


def release_store() -> None:
    """
    Fully shut down ChromaDB and release every OS file handle.

    MUST be called before deleting chroma_db/ on Windows (WinError 32).
    Safe on all platforms; idempotent when already released.

    Shutdown sequence
    -----------------
    1. _client._system.stop()
       Stops LocalSegmentManager (closes hnswlib mmap → data_level0.bin)
       and SqliteDB (closes sqlite3 connection → chroma.sqlite3 + WAL).
    2. _store = None  /  _client = None   — drop all Python references.
    3. gc.collect()   — run C-extension destructors immediately.
    4. time.sleep(0.3) — give Windows I/O manager time to flush handle table.
    """
    global _client, _store

    _store = None  # drop LangChain wrapper first

    if _client is not None:
        try:
            _client._system.stop()
        except Exception:
            try:
                _client.stop()          # chromadb < 0.4 fallback
            except Exception:
                pass
        _client = None

    gc.collect()
    time.sleep(0.3)


# ─────────────────────────────────────────────────────────────────────────────
# Write operations
# ─────────────────────────────────────────────────────────────────────────────

def add_documents(chunks: list[Document]) -> None:
    """
    Embed and persist a list of LangChain Documents into ChromaDB.

    Uses the shared PersistentClient so we never open a second client
    against the same directory (which would cause tenant-not-found errors).
    """
    cfg       = get_settings()
    embeddings = get_embedding_model()
    client    = _get_or_create_client()

    # Build a fresh Chroma wrapper tied to our managed client
    store = _build_store(client)
    store.add_documents(chunks)

    # Invalidate the cached wrapper so next get_store() reflects new counts
    invalidate_store()


# ─────────────────────────────────────────────────────────────────────────────
# Read / retrieval
# ─────────────────────────────────────────────────────────────────────────────

def get_retriever(search_type: str = "similarity"):
    """
    Return a LangChain retriever backed by the current vector store.
    Raises RuntimeError if the store is empty.
    """
    store = get_store()
    if store is None:
        raise RuntimeError("Vector store is empty — upload a PDF first.")

    cfg = get_settings()
    return store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": cfg.retriever_top_k},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_chunk_count() -> int:
    """Total number of embedded chunks currently in the collection."""
    try:
        client = _get_or_create_client()
        cfg    = get_settings()
        col    = client.get_or_create_collection(cfg.chroma_collection)
        return col.count()
    except Exception:
        return 0


def get_ingested_filenames() -> list[str]:
    """Sorted list of unique source filenames stored in collection metadata."""
    try:
        client = _get_or_create_client()
        cfg    = get_settings()
        col    = client.get_or_create_collection(cfg.chroma_collection)
        result = col.get(include=["metadatas"])
        names  = {
            Path(m.get("source", "")).name
            for m in result["metadatas"]
            if m.get("source")
        }
        return sorted(names)
    except Exception:
        return []


def db_is_ready() -> bool:
    """True if the collection contains at least one chunk."""
    return get_chunk_count() > 0