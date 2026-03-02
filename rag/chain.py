"""
app/rag/chain.py
────────────────
LangChain RAG chain construction.

Responsibilities
----------------
- Build the Groq LLM
- Compose the full LCEL retrieval-augmented generation chain
- Format retrieved docs with source attribution
- Return (chain, retriever) for callers that need both

Design notes
------------
The chain is built fresh per request (stateless).  This is intentional:
  • No shared mutable state between concurrent requests
  • LLM config changes take effect immediately without restart
  • Easy to swap prompt / LLM / retriever for A/B testing

The chain follows the LCEL pattern:
    question → {context: retriever | format_docs, question: passthrough}
             → prompt
             → llm
             → output_parser
             → str
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from core.config import get_settings
from rag.vectorstore import get_retriever

# ── Prompt template ───────────────────────────────────────────────────────────
# Edit the prompt here to change the LLM's behavior globally.
_RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a precise, helpful assistant.
Answer using ONLY the context provided below.
If the answer is not present in the context, respond with:
"I don't have enough information in the provided documents to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


# ── Document formatter ────────────────────────────────────────────────────────
def _format_docs(docs) -> str:
    """
    Render retrieved documents as a single context string.

    Each chunk is prefixed with its source file and page number so the LLM
    can reference them and so the caller can surface them to the user.
    """
    sections = []
    for doc in docs:
        source = Path(doc.metadata.get("source", "unknown")).name
        page   = doc.metadata.get("page", "?")
        sections.append(f"[Source: {source} | Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


# ── LLM factory ───────────────────────────────────────────────────────────────
def _build_llm() -> ChatGroq:
    cfg = get_settings()
    return ChatGroq(
        api_key=cfg.groq_api_key,
        model=cfg.groq_model,
        temperature=cfg.groq_temperature,
        max_tokens=cfg.groq_max_tokens,
    )


# ── Public API ────────────────────────────────────────────────────────────────
def build_rag_chain():
    """
    Assemble and return the complete RAG chain plus the retriever.

    Returns
    -------
    chain     : Runnable  –  chain.invoke(question) → str answer
    retriever : Retriever –  retriever.invoke(question) → list[Document]

    Usage
    -----
    chain, retriever = build_rag_chain()
    answer = chain.invoke("What is the document about?")
    sources = retriever.invoke("What is the document about?")
    """
    retriever = get_retriever()

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | _RAG_PROMPT
        | _build_llm()
        | StrOutputParser()
    )

    return chain, retriever