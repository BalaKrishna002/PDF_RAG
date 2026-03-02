"""
app/core/config.py
──────────────────
Central configuration using pydantic-settings.
All environment variables are read here — never scattered across the codebase.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment / .env file.
    Add new config here; never use os.getenv() elsewhere.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Groq LLM ──────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.2
    groq_max_tokens: int = 1024

    # ── HuggingFace Embeddings ─────────────────────────
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_device: str = "cpu"
    embed_normalize: bool = True

    # ── ChromaDB ──────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "pdf_rag"

    # ── Chunking ──────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_top_k: int = 4

    # ── File Storage ──────────────────────────────────
    upload_dir: str = "./uploads"
    max_upload_mb: int = 50

    # ── App ───────────────────────────────────────────
    app_title: str = "PDF RAG API"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Derived helpers ───────────────────────────────
    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached singleton — call get_settings() anywhere; .env is read only once.
    """
    return Settings()