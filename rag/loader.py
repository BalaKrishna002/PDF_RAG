"""
app/rag/loader.py
─────────────────
PDF loading and text chunking.

Responsibilities
----------------
- Load a PDF file from disk → list[Document]
- Split into overlapping chunks suitable for embedding
- Return metadata-enriched chunks ready for vectorstore.add_documents()

Extending
---------
To support more file types (DOCX, TXT, HTML …), add a new load_* function
and wire it into load_and_chunk() via a dispatch dict keyed on file extension.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import get_settings


def load_pdf(pdf_path: str) -> list[Document]:
    """
    Load every page of a PDF as a LangChain Document.

    Each Document carries metadata:
        source  – absolute path to the file
        page    – 0-based page index
    """
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into overlapping text chunks.

    chunk_size and chunk_overlap are read from Settings so they can be
    tuned via .env without modifying code.
    """
    cfg = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(documents)


def load_and_chunk(file_path: str) -> tuple[list[Document], dict]:
    """
    High-level helper: load → split → return (chunks, stats).

    Returns
    -------
    chunks : list[Document]
        Ready-to-embed text chunks.
    stats  : dict
        {"pages": int, "chunks": int} for logging / API responses.
    """
    path = Path(file_path)

    # Dispatch by extension — easy to extend for other file types
    _loaders = {
        ".pdf": load_pdf,
    }
    loader_fn = _loaders.get(path.suffix.lower())
    if loader_fn is None:
        raise ValueError(f"Unsupported file type: '{path.suffix}'")

    documents = loader_fn(file_path)
    chunks = chunk_documents(documents)

    stats = {"pages": len(documents), "chunks": len(chunks)}
    return chunks, stats