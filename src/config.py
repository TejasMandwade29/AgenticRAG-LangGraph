"""
Configuration module for the RAG chatbot.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"

# --- Vector Store ---
VECTOR_STORE = os.getenv("VECTOR_STORE", "pinecone").lower()  # "pinecone" or "chroma"
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-rag")

# --- Embeddings ---
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

# --- LLM ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # "groq" or "openai"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Chunking ---
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# --- Retrieval ---
TOP_K = 5

# Similarity normalization range — calibrated from 6 real queries against live ChromaDB.
# Raw cosine similarity distribution (all-MiniLM-L6-v2, ChromaDB L2→cosine):
#   Strong in-scope queries:  0.73–0.94
#   Moderate in-scope:        0.29–0.55
#   Weak/near-miss:           0.00–0.10
#   Truly out-of-scope:       0.00
# LOW=0.27 = natural in-scope floor; anything below is treated as no-signal (→ 0.0 normalized)
# HIGH=0.90 = slightly below observed max of 0.94 so top scores spread across [0.87, 1.0]
#             rather than bunching at exactly 1.0 for the single best chunk.
SIM_NORM_LOW: float = 0.27
SIM_NORM_HIGH: float = 0.90

# --- API ---
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# --- Chapter mapping (page ranges derived from PDF analysis) ---
CHAPTER_MAP = {
    1: {"title": "Introduction to Agentic AI", "pages": range(7, 17)},
    2: {"title": "Anatomy of an Agentic AI System", "pages": range(17, 29)},
    3: {"title": "Multi-Agent Systems", "pages": range(29, 37)},
    4: {"title": "Orchestrating Agentic AI Systems", "pages": range(37, 48)},
    5: {"title": "Your Readiness for Agentic AI", "pages": range(48, 54)},
    6: {"title": "Practical Applications of Agentic AI", "pages": range(54, 59)},
}


def get_chapter_for_page(page_num: int) -> tuple[int | None, str]:
    """Return (chapter_number, chapter_title) for a given 1-indexed page number."""
    for ch_num, ch_info in CHAPTER_MAP.items():
        if page_num in ch_info["pages"]:
            return ch_num, ch_info["title"]
    return None, "Front Matter / Back Matter"
