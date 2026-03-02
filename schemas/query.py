"""
app/schemas/query.py
─────────────────────
Pydantic models for the RAG query endpoint.
"""

from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """A single retrieved document chunk surfaced to the caller."""

    file: str    = Field(..., description="Source filename")
    page: int | str = Field(..., description="Page number (0-based) or '?' if unknown")
    snippet: str = Field(..., description="First 200 characters of the chunk text")


class QueryResponse(BaseModel):
    """Returned after a successful RAG query."""

    success: bool = True
    question: str  = Field(..., description="The original question")
    answer: str    = Field(..., description="LLM answer grounded in retrieved context")
    sources: list[SourceChunk] = Field(default_factory=list, description="Retrieved chunks used as context")
    elapsed: float = Field(..., description="Wall-clock seconds for the full RAG pipeline")