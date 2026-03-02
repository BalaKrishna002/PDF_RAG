"""
app/api/v1/router.py
──────────────────────
Aggregates all v1 sub-routers under the /api/v1 prefix.

To add a new resource (e.g. /api/v1/collections):
    1. Create app/api/v1/collections.py with an APIRouter
    2. Import it here and call router.include_router(...)
"""

from fastapi import APIRouter

from api.v1 import documents, query, system

router = APIRouter(prefix="/api/v1")

router.include_router(documents.router)
router.include_router(query.router)
router.include_router(system.router)