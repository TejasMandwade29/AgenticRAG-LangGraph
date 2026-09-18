"""
FastAPI application exposing the LangGraph RAG chatbot API.

Endpoints:
  POST /chat    — Process user question through the LangGraph state machine
  GET  /health  — Service health and configuration details
  GET  /        — Welcome and API overview
"""

import logging
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import (
    API_HOST,
    API_PORT,
    VECTOR_STORE,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    LLM_PROVIDER,
    GROQ_MODEL,
    OPENAI_MODEL,
)
from src.graph import run_graph, get_vectorstore, get_llm
from src.schemas import ChatRequest, ChatResponse, RetrievedChunk, GraphState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up components on startup (load vector store and LLM)."""
    logger.info("Starting up Agentic AI RAG Chatbot API...")
    try:
        # Pre-initialize vector store and LLM so first user query has low latency
        get_vectorstore()
        get_llm()
        logger.info("Vector store and LLM pre-loaded successfully.")
    except Exception as e:
        logger.warning("Warm-up warning (non-fatal, will retry on demand): %s", e)
    yield
    logger.info("Shutting down Agentic AI RAG Chatbot API.")


app = FastAPI(
    title="Agentic AI RAG Chatbot API",
    description=(
        "Production-quality RAG Chatbot powered by LangGraph, Vector DB, "
        "and Embeddings, grounded strictly in 'Agentic AI: An Executive's Guide' by Konverge AI."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend / Streamlit / web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_retrieved_chunks(documents: list) -> list[RetrievedChunk]:
    """Transform LangChain Document objects into API schema RetrievedChunk instances."""
    chunks = []
    for doc in documents:
        meta = doc.metadata or {}
        chunk = RetrievedChunk(
            text=doc.page_content,
            page_number=int(meta.get("page_number", 1)),
            section=str(meta.get("section_title", "Unknown")),
            similarity_score=float(meta.get("similarity_score", 0.0)),
            content_type=meta.get("content_type", "text"),
        )
        chunks.append(chunk)
    return chunks


@app.get("/", tags=["General"])
async def root() -> dict[str, Any]:
    """Welcome and status summary."""
    return {
        "message": "Welcome to the Agentic AI RAG Chatbot API",
        "status": "online",
        "documentation": "/docs",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health",
        },
    }


@app.get("/health", tags=["General"])
async def health_check() -> dict[str, Any]:
    """Health check endpoint providing runtime and configuration status."""
    active_model = GROQ_MODEL if LLM_PROVIDER == "groq" else OPENAI_MODEL
    return {
        "status": "healthy",
        "vector_store": VECTOR_STORE,
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": EMBEDDING_MODEL,
        "llm_provider": LLM_PROVIDER,
        "llm_model": active_model,
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Ask a question grounded in the Agentic AI eBook",
)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Execute the multi-step LangGraph workflow for the user's question:
    1. Retrieve relevant chunks with calibrated similarity scoring.
    2. Batched relevance grading with the LLM.
    3. Grounded answer generation (with diagram/table disclaimers if applicable).
    4. Hallucination verification and automated retry loop (max 3 attempts).
    5. Composite confidence calculation (retrieval + groundedness).
    """
    cleaned_query = request.question.strip()
    if not cleaned_query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty or whitespace only.",
        )

    logger.info("Received chat query: %r", cleaned_query)

    try:
        state: GraphState = run_graph(cleaned_query)
    except Exception as e:
        logger.exception("Error executing LangGraph pipeline for query: %r", cleaned_query)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error processing request: {str(e)}",
        )

    refused = state.get("refused", False)
    docs = state.get("documents", [])
    retrieved_chunks = [] if refused else _build_retrieved_chunks(docs)
    grounded = state.get("groundedness", 0.0) >= 0.5

    response = ChatResponse(
        answer=state.get("answer", ""),
        retrieved_chunks=retrieved_chunks,
        confidence=float(state.get("confidence", 0.0)),
        confidence_label=state.get("confidence_label", "Low"),
        grounded=grounded,
        refused=refused,
    )

    logger.info(
        "Chat response generated: refused=%s, confidence=%.3f (%s), chunks=%d",
        response.refused,
        response.confidence,
        response.confidence_label,
        len(response.retrieved_chunks),
    )
    return response


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
    )
