"""
app/main.py
────────────
FastAPI application factory.

Wiring order
------------
1. Create FastAPI instance with metadata from Settings
2. Mount static files
3. Register Jinja2 templates
4. Include API routers
5. Register the UI (Jinja2) route last so /docs stays at /api/v1
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.v1.router import router as api_v1_router
from core.config import get_settings
from rag import vectorstore as vs

# ── Settings ──────────────────────────────────────────────────────────────────
cfg = get_settings()

# ── App instance ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=cfg.app_title,
    version=cfg.app_version,
    description="Retrieval-Augmented Generation API — Groq × HuggingFace × ChromaDB",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Static files ──────────────────────────────────────────────────────────────
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── Templates ─────────────────────────────────────────────────────────────────
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# ── API routers ───────────────────────────────────────────────────────────────
app.include_router(api_v1_router)

# ── UI route ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui(request: Request):
    """Serve the Jinja2 single-page interface."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "db_ready": vs.db_is_ready(),
            "doc_count": vs.get_chunk_count(),
            "files": vs.get_ingested_filenames(),
        },
    )