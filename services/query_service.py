"""
app/services/query_service.py
──────────────────────────────
Orchestrates the RAG query pipeline.

Keeps route handlers free of LangChain details.
All LLM / retriever interactions live here.
"""

from __future__ import annotations

import time
from pathlib import Path

from core.config import get_settings
from core.exceptions import GroqKeyMissingError, QueryError, VectorStoreNotReadyError
from rag.chain import build_rag_chain
from rag import vectorstore as vs
from schemas.query import QueryResponse, SourceChunk


def answer_question(question: str) -> QueryResponse:
    """
    Run the full RAG pipeline for a user question.

    Steps
    -----
    1. Guard: DB ready? Groq key set?
    2. Build the LCEL chain (stateless per request)
    3. Invoke chain → answer string
    4. Fetch top-k sources for provenance display
    5. Return structured QueryResponse

    Parameters
    ----------
    question : str
        The user's natural-language question.

    Raises
    ------
    VectorStoreNotReadyError  – no documents in the index
    GroqKeyMissingError       – GROQ_API_KEY not configured
    QueryError                – any pipeline exception
    """
    cfg = get_settings()

    # ── Guards ─────────────────────────────────────────
    if not vs.db_is_ready():
        raise VectorStoreNotReadyError()

    if not cfg.groq_api_key:
        raise GroqKeyMissingError()

    # ── Pipeline ───────────────────────────────────────
    t0 = time.perf_counter()
    try:
        chain, retriever = build_rag_chain()
        answer   = chain.invoke(question)
        raw_docs = retriever.invoke(question)
    except Exception as exc:
        raise QueryError(str(exc)) from exc

    elapsed = round(time.perf_counter() - t0, 2)

    # ── Build source list ──────────────────────────────
    sources = [
        SourceChunk(
            file=Path(doc.metadata.get("source", "unknown")).name,
            page=doc.metadata.get("page", "?"),
            snippet=doc.page_content[:200],
        )
        for doc in raw_docs
    ]

    return QueryResponse(
        question=question,
        answer=answer,
        sources=sources,
        elapsed=elapsed,
    )