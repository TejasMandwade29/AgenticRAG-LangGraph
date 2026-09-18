"""
Pydantic models for the RAG chatbot API request and response,
and for the LangGraph state.
"""

from typing import Literal
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Annotated
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# API Schemas
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A single context chunk returned with the answer."""

    text: str = Field(description="The chunk text content.")
    page_number: int = Field(description="PDF page number (1-indexed).")
    section: str = Field(description="Section title detected in the chunk (e.g. '3.4 Challenges…').")
    similarity_score: float = Field(
        ge=0.0, le=1.0,
        description="Cosine similarity to the query, normalized to [0, 1]."
    )
    content_type: Literal["text", "diagram_fragment", "table_fragment"] = Field(
        default="text",
        description=(
            "'text' for normal prose/table with preserved structure; "
            "'diagram_fragment' for visual flowchart labels whose relationships are lost in extraction; "
            "'table_fragment' for multi-column table cells whose row-column mapping is ambiguous."
        ),
    )


class ChatRequest(BaseModel):
    """Incoming chat request."""

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="The user's question to the RAG chatbot.",
    )


class ChatResponse(BaseModel):
    """Full response from the RAG pipeline."""

    answer: str = Field(
        description="The generated answer, grounded in the source material."
    )
    retrieved_chunks: list[RetrievedChunk] = Field(
        description="The context chunks retrieved and used (or considered) for generation. "
                    "Empty on refusal."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Composite confidence score: "
            "0.6 × normalized_mean_similarity + 0.4 × groundedness_verdict. "
            "0.0 on refusal."
        ),
    )
    confidence_label: Literal["High", "Medium", "Low"] = Field(
        description="'High' (>0.75), 'Medium' (0.45–0.75), 'Low' (<0.45). Always 'Low' on refusal."
    )
    grounded: bool = Field(
        description="True if verify_groundedness returned 'fully_grounded' or 'partially_grounded'."
    )
    refused: bool = Field(
        description=(
            "True if the system refused to answer — either because no relevant chunks "
            "were found or because the answer failed groundedness checks after max retries."
        )
    )
    groundedness: float = Field(
        default=0.0,
        description="Groundedness verdict score (1.0 = fully grounded, 0.5 = partially, 0.0 = not grounded)",
    )
    retry_count: int = Field(
        default=1,
        description="Total generation attempts executed (1 = succeeded first try, 2+ = retried)",
    )


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    """
    The mutable state that flows through every node in the LangGraph pipeline.

    question         — user query, set at entry, never modified.
    documents        — retrieved chunks; grade_documents filters this in-place.
    retrieval_scores — raw normalized similarity scores, parallel to pre-grade documents list.
    answer           — set by generate / refuse.
    groundedness     — 1.0 (fully) / 0.5 (partial) / 0.0 (not grounded). Set by verify or refuse.
    retry_count      — incremented by generate on every call; refuse path fires when >= 3.
    refused          — set True by the refuse node.
    confidence       — composite score, set by compute_confidence or forced to 0.0 by refuse.
    confidence_label — "High" / "Medium" / "Low", set by compute_confidence or "Low" by refuse.
    """

    question: str
    documents: list[Document]
    retrieval_scores: list[float]
    answer: str
    groundedness: float
    retry_count: int
    refused: bool
    confidence: float
    confidence_label: str
