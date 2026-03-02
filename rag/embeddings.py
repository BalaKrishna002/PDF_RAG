"""
app/rag/embeddings.py
─────────────────────
HuggingFace embedding model — singleton, lazy-loaded.

Swap the model name in .env (EMBED_MODEL=...) to use any
sentence-transformers compatible model without touching code.
"""

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from core.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Return the singleton HuggingFaceEmbeddings instance.

    The model is downloaded on first call and cached for the process lifetime.
    lru_cache ensures we never load it twice even under concurrent requests.
    """
    cfg = get_settings()

    return HuggingFaceEmbeddings(
        model_name=cfg.embed_model,
        model_kwargs={"device": cfg.embed_device},
        encode_kwargs={"normalize_embeddings": cfg.embed_normalize},
    )