"""
Retriever module: loads the vector store and exposes retrieval with score normalization.

Design decisions:
  - ChromaDB: returns L2 distance (lower = more similar). For unit-normalized vectors
    (which HuggingFace embeddings produce when normalize_embeddings=True):
        cosine_similarity = 1 - (L2_distance² / 2)
    We convert to cosine similarity before returning.
  - Pinecone: returns cosine similarity directly (higher = more similar, range 0–1).
    No conversion needed.
  - Scores are then passed through normalize_similarity(), which rescales from the
    empirically observed raw range [SIM_LOW, SIM_HIGH] → [0.0, 1.0].
    The constants SIM_LOW/SIM_HIGH are set in config.py AFTER running calibrate_scores()
    on real queries. They are left as None here until that step is done.
"""

import json
import logging
from pathlib import Path

from langchain_core.documents import Document

from src.config import (
    VECTOR_STORE,
    CHROMA_PERSIST_DIR,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    TOP_K,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Similarity normalization constants
# These are set empirically in config.py after running calibrate_scores().
# Until then, the raw cosine similarity is returned as-is (identity mapping).
# ---------------------------------------------------------------------------
# Import from config; will be None until calibrated.
from src.config import SIM_NORM_LOW, SIM_NORM_HIGH


# ---------------------------------------------------------------------------
# Embedding function (shared between ingestion and retrieval)
# ---------------------------------------------------------------------------

def get_embedding_function():
    """Return the configured embedding function (matches ingest.py)."""
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)
    else:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


# ---------------------------------------------------------------------------
# Vector store loader (read-only, does NOT recreate the index)
# ---------------------------------------------------------------------------

def load_vectorstore():
    """
    Load an existing vector store for retrieval.
    Raises RuntimeError if the store does not exist yet (run ingest.py first).
    """
    if VECTOR_STORE == "chroma":
        return _load_chroma()
    elif VECTOR_STORE == "pinecone":
        return _load_pinecone()
    else:
        raise ValueError(f"Unknown VECTOR_STORE '{VECTOR_STORE}'. Use 'chroma' or 'pinecone'.")


def _load_chroma():
    from langchain_chroma import Chroma

    persist_dir = Path(CHROMA_PERSIST_DIR)
    if not persist_dir.exists():
        raise RuntimeError(
            f"ChromaDB not found at {CHROMA_PERSIST_DIR}. "
            "Run: python -m src.ingest --recreate"
        )
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="agentic_ai_ebook",
        embedding_function=get_embedding_function(),
    )


def _load_pinecone():
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore

    if not PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY not set. "
            "Add it to .env or set VECTOR_STORE=chroma for local testing."
        )
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        raise RuntimeError(
            f"Pinecone index '{PINECONE_INDEX_NAME}' not found. "
            "Run: python -m src.ingest --recreate"
        )
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=get_embedding_function(),
    )


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def chroma_distance_to_cosine(l2_distance: float) -> float:
    """
    Convert ChromaDB L2 distance to cosine similarity.
    For unit-normalized vectors: cosine_sim = 1 - (L2² / 2).
    Clamped to [0, 1].
    """
    cosine_sim = 1.0 - (l2_distance ** 2) / 2.0
    return max(0.0, min(1.0, cosine_sim))


def normalize_similarity(raw_cosine: float) -> float:
    """
    Rescale a raw cosine similarity from the empirically observed range
    [SIM_NORM_LOW, SIM_NORM_HIGH] → [0.0, 1.0].

    If SIM_NORM_LOW/HIGH have not been calibrated yet (None), returns raw_cosine
    unchanged so the pipeline still works — calibration just improves the metric.
    """
    low = SIM_NORM_LOW
    high = SIM_NORM_HIGH

    if low is None or high is None or high <= low:
        # Not calibrated yet — pass through
        return max(0.0, min(1.0, raw_cosine))

    clamped = max(low, min(high, raw_cosine))
    return (clamped - low) / (high - low)


# ---------------------------------------------------------------------------
# Core retrieval function
# ---------------------------------------------------------------------------

def retrieve_with_scores(
    vectorstore,
    query: str,
    k: int = TOP_K,
) -> list[tuple[Document, float, float]]:
    """
    Retrieve top-k documents and return (Document, raw_cosine_sim, normalized_sim).

    Both raw and normalized scores are returned so the caller can:
    - Use normalized_sim for the confidence formula
    - Log raw_cosine_sim during the calibration phase to set SIM_NORM_LOW/HIGH

    Args:
        vectorstore: loaded ChromaDB or Pinecone vector store
        query: user question
        k: number of chunks to retrieve

    Returns:
        List of (Document, raw_cosine_similarity, normalized_similarity) tuples,
        sorted descending by raw_cosine_similarity.
    """
    raw_results = vectorstore.similarity_search_with_score(query, k=k)

    results = []
    for doc, score in raw_results:
        if VECTOR_STORE == "chroma":
            # score is L2 distance → convert to cosine similarity
            raw_cosine = chroma_distance_to_cosine(score)
        else:
            # Pinecone returns cosine similarity directly
            raw_cosine = float(score)

        normalized = normalize_similarity(raw_cosine)
        doc.metadata["similarity_score"] = round(normalized, 4)
        doc.metadata["raw_cosine"] = round(raw_cosine, 4)
        results.append((doc, raw_cosine, normalized))

    # Sort descending by raw cosine (already sorted by ChromaDB/Pinecone but be explicit)
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Calibration helper: log raw scores for 5-6 queries to set normalization range
# ---------------------------------------------------------------------------

def calibrate_scores(vectorstore, queries: list[str], k: int = TOP_K) -> dict:
    """
    Run a set of representative queries, log their raw cosine similarity scores,
    and compute statistics to choose SIM_NORM_LOW and SIM_NORM_HIGH.

    Call this ONCE after building the index, before finalizing compute_confidence.
    The returned stats dict shows min, max, mean, and per-query scores.
    """
    all_scores = []
    per_query = {}

    for query in queries:
        results = retrieve_with_scores(vectorstore, query, k=k)
        scores = [r[1] for r in results]   # raw cosine similarities
        per_query[query] = {
            "scores": [round(s, 4) for s in scores],
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "mean": round(sum(scores) / len(scores), 4),
        }
        all_scores.extend(scores)
        logger.info("Query: %r  scores: %s", query[:60], scores)

    global_min = min(all_scores)
    global_max = max(all_scores)
    global_mean = sum(all_scores) / len(all_scores)

    stats = {
        "global_min": round(global_min, 4),
        "global_max": round(global_max, 4),
        "global_mean": round(global_mean, 4),
        "total_scores": len(all_scores),
        "per_query": per_query,
        "recommendation": {
            "SIM_NORM_LOW": round(global_min, 2),
            "SIM_NORM_HIGH": round(global_max, 2),
            "note": (
                "Set these in src/config.py. "
                "Consider tightening SIM_NORM_HIGH slightly below global_max "
                "to avoid scores bunching at 1.0 for the best match."
            ),
        },
    }
    return stats


# ---------------------------------------------------------------------------
# CLI: run calibration and print results
# ---------------------------------------------------------------------------

CALIBRATION_QUERIES = [
    "How does Agentic AI differ from RPA?",
    "Explain the six core pillars from perception to execution.",
    "What are the challenges of multi-agent systems and their mitigation strategies?",
    "What are the four readiness levels in the industry-specific AI readiness analysis?",
    "What impact did the Factory 4.0 use case deliver?",
    "Who is the Prime Minister of India?",   # Out-of-scope query — expect low scores
]


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("Loading vector store...")
    vs = load_vectorstore()

    print(f"\nRunning calibration on {len(CALIBRATION_QUERIES)} queries...\n")
    stats = calibrate_scores(vs, CALIBRATION_QUERIES)

    print("\n" + "=" * 70)
    print("SCORE CALIBRATION RESULTS")
    print("=" * 70)
    print(json.dumps(stats, indent=2))

    print("\n" + "=" * 70)
    print("RECOMMENDATION — set these in src/config.py:")
    print(f"  SIM_NORM_LOW  = {stats['recommendation']['SIM_NORM_LOW']}")
    print(f"  SIM_NORM_HIGH = {stats['recommendation']['SIM_NORM_HIGH']}")
    print("=" * 70)
