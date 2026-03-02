"""
app/core/exceptions.py
──────────────────────
Domain-specific exceptions mapped to HTTP status codes.
Raise these in services/rag; FastAPI handlers convert them automatically.
"""

from fastapi import HTTPException, status


class VectorStoreNotReadyError(HTTPException):
    """Raised when a query arrives but ChromaDB has no documents yet."""

    def __init__(self, detail: str = "No documents ingested. Upload a PDF first."):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)


class GroqKeyMissingError(HTTPException):
    """Raised when GROQ_API_KEY is not configured."""

    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GROQ_API_KEY is not set. Configure it in your .env file.",
        )


class UnsupportedFileTypeError(HTTPException):
    """Raised when the uploaded file is not a PDF."""

    def __init__(self, filename: str):
        super().__init__(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"'{filename}' is not a PDF. Only .pdf files are accepted.",
        )


class FileTooLargeError(HTTPException):
    """Raised when the uploaded file exceeds the configured size limit."""

    def __init__(self, limit_mb: int):
        super().__init__(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {limit_mb} MB upload limit.",
        )


class IngestError(HTTPException):
    """Raised when the PDF ingest pipeline fails unexpectedly."""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingest failed: {detail}",
        )


class QueryError(HTTPException):
    """Raised when the RAG query pipeline fails unexpectedly."""

    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {detail}",
        )