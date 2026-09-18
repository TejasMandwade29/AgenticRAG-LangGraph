"""
LangGraph RAG pipeline: 6-node state machine.

Topology:
    START
      → retrieve
      → grade_documents
      → [route_after_grading]
           ├─ no relevant docs ──────────────────────────────────→ refuse → END
           └─ relevant docs found
                → generate
                → verify_groundedness
                → [route_after_verification]
                     ├─ grounded                              → compute_confidence → END
                     ├─ not grounded + retry_count < 3        → generate (retry)
                     └─ not grounded + retry_count >= 3       → refuse → END

Key design decisions:
  - Batched grading: single LLM call for all chunks (plan fix #5).
  - retry_count increments on every generate call; max 3 attempts before refuse.
  - Groq LLM wrapped with tenacity exponential backoff for rate-limit safety.
  - Diagram/table disclaimer injected into generation prompt when any retrieved chunk
    has content_type != 'text'.
  - confidence = 0.6 × mean(normalized retrieval scores) + 0.4 × groundedness_verdict
    using SIM_NORM_LOW/HIGH calibrated from real score data.
"""

import json
import logging
import re
import time
from functools import lru_cache
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from src.config import (
    LLM_PROVIDER,
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
    SIM_NORM_LOW,
    SIM_NORM_HIGH,
)
from src.schemas import GraphState
from src.prompts import (
    GRADE_PROMPT,
    GENERATE_PROMPT,
    VERIFY_PROMPT,
    format_context,
    format_numbered_chunks,
    get_diagram_disclaimer,
)
from src.retriever import load_vectorstore, retrieve_with_scores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_RETRIES = 3          # Max generate calls before forcing refuse
CONFIDENCE_BUCKET_HIGH = 0.75
CONFIDENCE_BUCKET_LOW  = 0.45

REFUSE_ANSWER = (
    "I'm sorry, but I can only answer questions based on the content of the "
    "\"Agentic AI: An Executive's Guide\" eBook by Konverge AI. "
    "Your question appears to be outside the scope of that material, or "
    "I was unable to find a sufficiently grounded answer in the source text."
)

# ---------------------------------------------------------------------------
# LLM initialization
# ---------------------------------------------------------------------------

def _build_llm():
    """Build and return the configured LLM client. Called once at module init."""
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY not set. Add it to .env or set LLM_PROVIDER=openai."
            )
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.0,
            max_tokens=450,
            max_retries=0,  # We handle retries ourselves via tenacity below
        )
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY not set.")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=OPENAI_MODEL,
            temperature=0.0,
            max_tokens=750,
            max_retries=0,
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use 'groq' or 'openai'.")


# LLM and vector store are lazy-initialized on first use to avoid import-time errors
# (e.g. missing API keys during test runs that don't need the LLM).
_llm = None
_vectorstore = None


def get_llm():
    """Lazy-load LLM, cached for the process lifetime."""
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


def get_vectorstore():
    """Lazy-load vector store, cached for the process lifetime."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = load_vectorstore()
    return _vectorstore


# ---------------------------------------------------------------------------
# LLM call with exponential backoff (handles Groq rate limits)
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=5, max=30),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _llm_invoke(prompt: str, max_tokens: int = None) -> str:
    """
    Invoke the LLM with a raw string prompt.
    Retries up to 5 times with exponential backoff on any
    exception — primarily for Groq rate-limit (429) errors.
    Returns the response text as a plain string.
    """
    llm = get_llm()
    kwargs = {}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = llm.invoke([HumanMessage(content=prompt)], **kwargs)
    return response.content


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    """
    Robustly parse a JSON object from LLM output.
    Handles:
      - Raw JSON:         '{"key": "value"}'
      - Markdown fences:  '```json\\n{"key": "value"}\\n```'
      - Leading/trailing whitespace or prose
    Raises ValueError if no valid JSON object is found.
    """
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # Try to find the first {...} block in the response
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: try to parse the whole stripped text
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Could not parse JSON from LLM response: {text!r}") from e


# ---------------------------------------------------------------------------
# Node: retrieve
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> dict:
    """
    Retrieve top-k chunks from the vector store.
    Reads:  question
    Writes: documents, retrieval_scores (normalized cosine similarity)
    """
    question = state["question"]
    vs = get_vectorstore()

    results = retrieve_with_scores(vs, question, k=TOP_K)

    documents = [doc for doc, _raw, _norm in results]
    # Store normalized scores for confidence computation
    retrieval_scores = [norm for _doc, _raw, norm in results]

    logger.info(
        "retrieve: %d chunks, scores=%s",
        len(documents),
        [round(s, 3) for s in retrieval_scores],
    )
    return {"documents": documents, "retrieval_scores": retrieval_scores}


# ---------------------------------------------------------------------------
# Node: grade_documents (batched — single LLM call)
# ---------------------------------------------------------------------------

def grade_documents(state: GraphState) -> dict:
    """
    Filter retrieved documents to those relevant to the question.
    Uses a single batched LLM call returning a JSON array of relevant indices.
    Reads:  question, documents
    Writes: documents (filtered to relevant only)
    """
    question = state["question"]
    documents = state["documents"]

    if not documents:
        return {"documents": []}

    numbered = format_numbered_chunks(documents)
    prompt = GRADE_PROMPT.format(question=question, numbered_chunks=numbered)

    try:
        raw_response = _llm_invoke(prompt, max_tokens=60)
        parsed = _parse_json(raw_response)
        relevant_indices = parsed.get("relevant_indices", [])

        # Validate: must be a list of integers within range
        relevant_indices = [
            int(i) for i in relevant_indices
            if isinstance(i, (int, float)) and 0 <= int(i) < len(documents)
        ]
    except Exception as e:
        logger.warning("grade_documents: JSON parse failed (%s) — keeping all chunks", e)
        # Conservative fallback: keep all chunks rather than incorrectly refusing
        relevant_indices = list(range(len(documents)))

    relevant_docs = [documents[i] for i in relevant_indices]

    logger.info(
        "grade_documents: %d/%d chunks retained (indices=%s)",
        len(relevant_docs), len(documents), relevant_indices,
    )
    return {"documents": relevant_docs}


# ---------------------------------------------------------------------------
# Conditional edge 1: route_after_grading
# ---------------------------------------------------------------------------

def route_after_grading(state: GraphState) -> str:
    """
    Route based on whether any relevant documents were found.
      len(documents) == 0  →  refuse  (no context, don't generate)
      len(documents) >= 1  →  generate
    """
    if len(state["documents"]) == 0:
        logger.info("route_after_grading → refuse (no relevant documents)")
        return "refuse"
    logger.info("route_after_grading → generate (%d docs)", len(state["documents"]))
    return "generate"


# ---------------------------------------------------------------------------
# Node: generate
# ---------------------------------------------------------------------------

def generate(state: GraphState) -> dict:
    """
    Generate an answer from the relevant context chunks.
    Reads:  question, documents, retry_count
    Writes: answer, retry_count (incremented)

    retry_count tracks total generate calls:
      After 1st call: retry_count = 1
      After 2nd call: retry_count = 2
      After 3rd call: retry_count = 3  → next routing sends to refuse
    """
    question  = state["question"]
    documents = state["documents"]
    retry_count = state.get("retry_count", 0)

    context     = format_context(documents)
    disclaimer  = get_diagram_disclaimer(documents)

    prompt = GENERATE_PROMPT.format(
        question=question,
        formatted_context=context,
        diagram_disclaimer=disclaimer,
    )

    if retry_count > 0:
        logger.info("generate: retry attempt %d/%d", retry_count + 1, MAX_RETRIES)
    else:
        logger.info("generate: first attempt")

    answer = _llm_invoke(prompt, max_tokens=450)
    new_retry_count = retry_count + 1

    logger.info("generate: produced %d-char answer (retry_count now %d)", len(answer), new_retry_count)
    return {"answer": answer, "retry_count": new_retry_count}


# ---------------------------------------------------------------------------
# Node: verify_groundedness
# ---------------------------------------------------------------------------

def verify_groundedness(state: GraphState) -> dict:
    """
    Verify that the generated answer is grounded in the retrieved context.
    Reads:  documents, answer
    Writes: groundedness  (1.0 = fully / 0.5 = partially / 0.0 = not grounded)
    """
    documents = state["documents"]
    answer    = state["answer"]

    context = format_context(documents)
    prompt  = VERIFY_PROMPT.format(formatted_context=context, answer=answer)

    SCORE_MAP = {
        "fully_grounded":    1.0,
        "partially_grounded": 0.5,
        "not_grounded":       0.0,
    }

    try:
        raw_response = _llm_invoke(prompt, max_tokens=40)
        parsed  = _parse_json(raw_response)
        verdict = parsed.get("verdict", "not_grounded")
        score   = SCORE_MAP.get(verdict, 0.0)
    except Exception as e:
        logger.warning("verify_groundedness: parse failed (%s) — defaulting to not_grounded", e)
        verdict = "not_grounded"
        score   = 0.0

    logger.info("verify_groundedness: verdict=%r, score=%.1f", verdict, score)
    return {"groundedness": score}


# ---------------------------------------------------------------------------
# Conditional edge 2: route_after_verification
# ---------------------------------------------------------------------------

def route_after_verification(state: GraphState) -> str:
    """
    Route based on groundedness verdict and retry budget.
      groundedness >= 0.5            →  compute_confidence (accept answer)
      groundedness < 0.5
        AND retry_count < 3          →  generate (retry with same context)
        AND retry_count >= 3         →  refuse   (exhausted retries)
    """
    groundedness = state["groundedness"]
    retry_count  = state["retry_count"]

    if groundedness >= 0.5:
        logger.info("route_after_verification → compute_confidence (grounded=%.1f)", groundedness)
        return "compute_confidence"

    if retry_count < MAX_RETRIES:
        logger.info(
            "route_after_verification → generate (retry %d/%d, groundedness=%.1f)",
            retry_count, MAX_RETRIES, groundedness,
        )
        return "generate"

    logger.info(
        "route_after_verification → refuse (exhausted %d retries, groundedness=%.1f)",
        MAX_RETRIES, groundedness,
    )
    return "refuse"


# ---------------------------------------------------------------------------
# Node: compute_confidence
# ---------------------------------------------------------------------------

def compute_confidence(state: GraphState) -> dict:
    """
    Compute the composite confidence score.
    Formula: confidence = 0.6 × mean(retrieval_scores) + 0.4 × groundedness
    Where retrieval_scores are already normalized to [0, 1] by the retriever.

    Reads:  retrieval_scores, groundedness
    Writes: confidence, confidence_label
    """
    scores       = state.get("retrieval_scores", [])
    groundedness = state.get("groundedness", 0.0)

    if scores:
        mean_sim = sum(scores) / len(scores)
    else:
        mean_sim = 0.0

    confidence = 0.6 * mean_sim + 0.4 * groundedness
    confidence = round(max(0.0, min(1.0, confidence)), 4)

    if confidence > CONFIDENCE_BUCKET_HIGH:
        label = "High"
    elif confidence >= CONFIDENCE_BUCKET_LOW:
        label = "Medium"
    else:
        label = "Low"

    logger.info(
        "compute_confidence: mean_sim=%.4f, groundedness=%.1f, confidence=%.4f (%s)",
        mean_sim, groundedness, confidence, label,
    )
    return {"confidence": confidence, "confidence_label": label}


# ---------------------------------------------------------------------------
# Node: refuse
# ---------------------------------------------------------------------------

def refuse(state: GraphState) -> dict:
    """
    Emit a refusal response.
    Triggered when: no relevant docs found, OR answer failed groundedness checks after max retries.
    Writes: answer, refused, groundedness, confidence, confidence_label
    """
    logger.info("refuse: emitting refusal response")
    return {
        "answer":           REFUSE_ANSWER,
        "refused":          True,
        "groundedness":     0.0,
        "confidence":       0.0,
        "confidence_label": "Low",
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph():
    """Compile the LangGraph state machine. Called once at module level."""
    builder = StateGraph(GraphState)

    # Register nodes
    builder.add_node("retrieve",             retrieve)
    builder.add_node("grade_documents",      grade_documents)
    builder.add_node("generate",             generate)
    builder.add_node("verify_groundedness",  verify_groundedness)
    builder.add_node("compute_confidence",   compute_confidence)
    builder.add_node("refuse",               refuse)

    # Edges
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "refuse": "refuse"},
    )
    builder.add_edge("generate", "verify_groundedness")
    builder.add_conditional_edges(
        "verify_groundedness",
        route_after_verification,
        {
            "compute_confidence": "compute_confidence",
            "generate":           "generate",
            "refuse":             "refuse",
        },
    )
    builder.add_edge("compute_confidence", END)
    builder.add_edge("refuse",             END)

    return builder.compile()


# Compiled graph — ready to invoke
graph = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_graph(question: str) -> dict:
    """
    Run the full RAG pipeline for a user question.

    Args:
        question: The user's query string.

    Returns:
        The final GraphState dict with all fields populated:
          answer, retrieved_chunks, confidence, confidence_label,
          grounded, refused.
    """
    initial_state: GraphState = {
        "question":         question,
        "documents":        [],
        "retrieval_scores": [],
        "answer":           "",
        "groundedness":     0.0,
        "retry_count":      0,
        "refused":          False,
        "confidence":       0.0,
        "confidence_label": "Low",
    }
    return graph.invoke(initial_state)


# ---------------------------------------------------------------------------
# CLI: smoke test — run two queries and print state
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    test_queries = [
        "How does Agentic AI differ from RPA and traditional LLMs?",
        "Who is the Prime Minister of India?",  # Expected: refusal
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"QUERY: {q}")
        print(f"{'='*70}")

        result = run_graph(q)

        print(f"  refused:           {result.get('refused')}")
        print(f"  confidence:        {result.get('confidence')} ({result.get('confidence_label')})")
        print(f"  groundedness:      {result.get('groundedness')}")
        print(f"  retry_count:       {result.get('retry_count')}")
        print(f"  chunks used:       {len(result.get('documents', []))}")
        print(f"\nANSWER:\n{result.get('answer')}")
