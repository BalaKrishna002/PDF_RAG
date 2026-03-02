"""
app/api/v1/documents.py
────────────────────────
Document ingest endpoints.

Routes are thin: validate HTTP concerns, delegate to services, return schemas.
No business logic lives here.
"""

from fastapi import APIRouter, File, UploadFile

from schemas.document import IngestResponse
from services import document_service

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=IngestResponse,
    summary="Upload and ingest a PDF",
    description=(
        "Upload a PDF file. The file is chunked, embedded via HuggingFace, "
        "and stored in ChromaDB. Subsequent uploads append to the same index."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
) -> IngestResponse:
    return await document_service.ingest_document(file)