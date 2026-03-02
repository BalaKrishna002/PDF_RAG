"""
api/v1/system.py
──────────────────────
System / ops endpoints: status check and database reset.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from schemas.document import StatusResponse
from services import document_service

router = APIRouter(prefix="/system", tags=["System"])


@router.get(
    "/status",
    response_model=StatusResponse,
    summary="Vector store health check",
)
def status() -> StatusResponse:
    """Return current state of the ChromaDB index and service config."""
    return document_service.get_status()


@router.delete(
    "/reset",
    summary="Wipe the database",
    description="Permanently delete all vectors and uploaded files. Cannot be undone.",
)
def reset() -> JSONResponse:
    document_service.reset_database()
    return JSONResponse({"success": True, "message": "Database cleared successfully."})