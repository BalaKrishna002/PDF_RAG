"""
app/api/v1/query.py
────────────────────
RAG query endpoint.

Accepts both JSON body and form-encoded data so the Jinja2 UI
can POST from a plain HTML form as well as fetch() JSON calls.
"""

from fastapi import APIRouter, Form

from schemas.query import QueryResponse
from services import query_service

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question",
    description=(
        "Send a natural-language question. The RAG pipeline retrieves relevant "
        "document chunks from ChromaDB and sends them as context to the Groq LLM."
    ),
)
def ask_question(
    question: str = Form(..., description="Natural-language question"),
) -> QueryResponse:
    return query_service.answer_question(question)