"""
Streamlit Web Interface for the Agentic AI RAG Chatbot.

Provides:
  - Conversational chat interface with message history
  - Dynamic display of Answer, Confidence badge (High / Medium / Low), and Grounding status
  - Expandable context chunks with PDF page citations, sections, and similarity scores
  - Prominent visual warning tags for diagram and table fragments
  - Sidebar with eBook metadata, pipeline architecture stats, and quick sample queries
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import httpx
from src.config import (
    API_HOST,
    API_PORT,
    LLM_PROVIDER,
    GROQ_MODEL,
    OPENAI_MODEL,
    VECTOR_STORE,
    EMBEDDING_MODEL,
    TOP_K,
)
from src.graph import run_graph
from src.schemas import ChatResponse, RetrievedChunk

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic AI Executive Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Styling ---
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-high {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #86EFAC;
    }
    .badge-medium {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #FDE047;
    }
    .badge-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #FCA5A5;
    }
    .badge-fragment {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid #FDBA74;
    }
    .chunk-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def query_rag(question: str) -> dict:
    """Query either via the FastAPI backend or fallback to direct graph execution."""
    api_url = f"http://{API_HOST if API_HOST != '0.0.0.0' else '127.0.0.1'}:{API_PORT}/chat"
    try:
        response = httpx.post(api_url, json={"question": question}, timeout=60.0)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass

    # Direct in-process fallback
    state = run_graph(question)
    refused = state.get("refused", False)
    docs = state.get("documents", [])
    chunks = []
    if not refused:
        for d in docs:
            m = d.metadata or {}
            chunks.append({
                "text": d.page_content,
                "page_number": m.get("page_number", 1),
                "section": m.get("section_title", "Unknown"),
                "similarity_score": m.get("similarity_score", 0.0),
                "content_type": m.get("content_type", "text"),
            })
    return {
        "answer": state.get("answer", ""),
        "retrieved_chunks": chunks,
        "confidence": state.get("confidence", 0.0),
        "confidence_label": state.get("confidence_label", "Low"),
        "grounded": state.get("groundedness", 0.0) >= 0.5,
        "refused": refused,
    }


# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_query" not in st.session_state:
    st.session_state.input_query = ""


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("Knowledge Base")
    st.markdown(
        """
        **Book**: *Agentic AI: An Executive's Guide*  
        **Publisher**: Konverge AI  
        **Corpus**: 60 pages · 6 Chapters · 125 Chunks  
        """
    )
    st.divider()

    st.subheader("⚙️ System Status")
    active_model = GROQ_MODEL if LLM_PROVIDER == "groq" else OPENAI_MODEL
    st.markdown(f"- **LLM**: `{active_model}` ({LLM_PROVIDER.upper()})")
    st.markdown(f"- **Vector DB**: `{VECTOR_STORE.upper()}`")
    st.markdown(f"- **Embeddings**: `{EMBEDDING_MODEL}`")
    st.markdown(f"- **Top-K Chunks**: `{TOP_K}`")
    st.divider()

    st.subheader("💡 Sample Questions")
    sample_queries = [
        "How does Agentic AI differ from RPA and traditional LLMs?",
        "Explain the six core pillars from perception to execution.",
        "What are the challenges of multi-agent systems and their mitigation strategies?",
        "What are the four readiness levels in the industry-specific AI readiness analysis?",
        "What impact did the Factory 4.0 use case deliver?",
        "Who is the Prime Minister of India?",
    ]

    for sq in sample_queries:
        if st.button(sq, key=f"btn_{sq}", use_container_width=True):
            st.session_state.pending_query = sq


# --- Header ---
st.markdown('<div class="main-title">🤖 Agentic AI Executive Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Multi-stage LangGraph RAG with relevance grading, hallucination verification, and strict grounding.</div>',
    unsafe_allow_html=True,
)


# --- Render Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            meta = msg["metadata"]
            label = meta.get("confidence_label", "Low")
            badge_class = f"badge-{label.lower()}"
            conf_val = meta.get("confidence", 0.0)
            grounded_txt = "✅ Grounded" if meta.get("grounded") else "❌ Not Grounded / Refused"

            st.markdown(
                f"""
                <div style="margin-top: 8px; margin-bottom: 8px; display: flex; gap: 10px; align-items: center;">
                    <span class="{badge_class}">Confidence: {conf_val:.1%} ({label})</span>
                    <span style="font-size: 0.85rem; color: #475569; font-weight: 500;">{grounded_txt}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            chunks = meta.get("retrieved_chunks", [])
            if chunks:
                with st.expander(f"📚 Retrieved Context Chunks ({len(chunks)})"):
                    for i, ch in enumerate(chunks):
                        ctype = ch.get("content_type", "text")
                        fragment_badge = (
                            f'<span class="badge-fragment">⚠️ {ctype.replace("_", " ").title()}</span>'
                            if ctype != "text" else ""
                        )
                        st.markdown(
                            f"""
                            <div class="chunk-container">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                                    <b>Chunk #{i+1} | Page {ch.get('page_number')} | Section: {ch.get('section')}</b>
                                    <div>{fragment_badge} <span style="color:#0284C7; font-weight:600;">Score: {ch.get('similarity_score', 0):.2f}</span></div>
                                </div>
                                <div style="font-size: 0.9rem; color: #334155; white-space: pre-wrap;">{ch.get('text')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )


# --- Handle Input ---
query_to_process = None

if "pending_query" in st.session_state and st.session_state.pending_query:
    query_to_process = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    user_input = st.chat_input("Ask a question about the Agentic AI eBook...")
    if user_input:
        query_to_process = user_input

if query_to_process:
    # 1. Append user message
    st.session_state.messages.append({"role": "user", "content": query_to_process})
    with st.chat_message("user"):
        st.markdown(query_to_process)

    # 2. Query RAG
    with st.chat_message("assistant"):
        with st.spinner("Analyzing knowledge base with LangGraph..."):
            res = query_rag(query_to_process)

        answer_text = res.get("answer", "No response received.")
        st.markdown(answer_text)

        label = res.get("confidence_label", "Low")
        conf_val = res.get("confidence", 0.0)
        badge_class = f"badge-{label.lower()}"
        grounded_txt = "✅ Grounded" if res.get("grounded") else "❌ Not Grounded / Refused"

        st.markdown(
            f"""
            <div style="margin-top: 8px; margin-bottom: 8px; display: flex; gap: 10px; align-items: center;">
                <span class="{badge_class}">Confidence: {conf_val:.1%} ({label})</span>
                <span style="font-size: 0.85rem; color: #475569; font-weight: 500;">{grounded_txt}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        chunks = res.get("retrieved_chunks", [])
        if chunks:
            with st.expander(f"📚 Retrieved Context Chunks ({len(chunks)})"):
                for i, ch in enumerate(chunks):
                    ctype = ch.get("content_type", "text")
                    fragment_badge = (
                        f'<span class="badge-fragment">⚠️ {ctype.replace("_", " ").title()}</span>'
                        if ctype != "text" else ""
                    )
                    st.markdown(
                        f"""
                        <div class="chunk-container">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                                <b>Chunk #{i+1} | Page {ch.get('page_number')} | Section: {ch.get('section')}</b>
                                <div>{fragment_badge} <span style="color:#0284C7; font-weight:600;">Score: {ch.get('similarity_score', 0):.2f}</span></div>
                            </div>
                            <div style="font-size: 0.9rem; color: #334155; white-space: pre-wrap;">{ch.get('text')}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # 3. Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "metadata": res,
        })
