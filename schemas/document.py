"""
app/schemas/document.py
────────────────────────
Pydantic models for the document ingest endpoint.

Keeping schemas separate from routes makes them reusable across
API versions and easy to evolve independently.
"""

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    """Returned after a successful PDF ingest."""

    success: bool = True
    filename: str = Field(..., description="Original filename of the uploaded PDF")
    pages: int    = Field(..., description="Number of pages in the PDF")
    chunks: int   = Field(..., description="Number of text chunks stored")
    elapsed: float = Field(..., description="Wall-clock seconds for the ingest pipeline")
    total_chunks: int = Field(..., description="Total chunks in the collection after ingest")
    ingested_files: list[str] = Field(default_factory=list, description="All files currently in the index")


class StatusResponse(BaseModel):
    """Vector store health check."""

    db_ready: bool
    total_chunks: int
    ingested_files: list[str]
    groq_key_configured: bool