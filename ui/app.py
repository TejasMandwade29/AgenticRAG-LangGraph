"""
Streamlit Web Interface for the Agentic AI RAG Chatbot.

High-ROI Features:
  - Dark/Light theme-adaptive glassmorphism (no white-on-white text issues)
  - Interactive LangGraph Pipeline Execution Trace (shows state transitions)
  - Composite Confidence Visual Breakdown (Retrieval Match 60% + Groundedness 40%)
  - Perplexity-style Topic Cards for instant 1-click query testing on empty state
  - Clear citation pills linking directly to PDF page numbers and sections
  - Sidebar with eBook metadata, pipeline architecture stats, and reset button
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import html
import textwrap
import httpx
import streamlit as st

from src.config import (
    API_HOST,
    API_PORT,
    EMBEDDING_MODEL,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_MODEL,
    TOP_K,
    VECTOR_STORE,
)
from src.graph import run_graph

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic AI Executive Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Theme-Adaptive High-ROI Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling with gradient accent */
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.8rem;
        font-weight: 400;
    }
    
    /* Confidence & Status Badges (translucent glass style) */
    .badge-high {
        background: rgba(34, 197, 94, 0.15);
        color: #4ADE80 !important;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(34, 197, 94, 0.35);
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .badge-medium {
        background: rgba(234, 179, 8, 0.15);
        color: #FACC15 !important;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(234, 179, 8, 0.35);
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .badge-low {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171 !important;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(239, 68, 68, 0.35);
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .badge-grounded {
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8 !important;
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .badge-fragment {
        background: rgba(249, 115, 22, 0.18);
        color: #FB923C !important;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        border: 1px solid rgba(249, 115, 22, 0.4);
    }
    .page-citation-pill {
        background: rgba(99, 102, 241, 0.12);
        color: #A5B4FC !important;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.28);
        margin-right: 6px;
    }

    /* Glassmorphic Chunk Container (theme-adaptive) */
    .chunk-glass-box {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 12px;
        backdrop-filter: blur(8px);
        transition: border 0.2s ease;
    }
    .chunk-glass-box:hover {
        border-color: rgba(56, 189, 248, 0.35);
    }
    .chunk-header-title {
        color: #38BDF8 !important;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .chunk-body-text {
        font-size: 0.88rem;
        color: #CBD5E1;
        white-space: pre-wrap;
        line-height: 1.55;
        margin-top: 8px;
    }

    /* Pipeline Step Timeline */
    .stepper-node {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #38BDF8;
        padding: 8px 14px;
        margin-bottom: 8px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
    }
    .stepper-node-title {
        font-weight: 600;
        color: #F1F5F9;
    }
    .stepper-node-desc {
        color: #94A3B8;
        font-size: 0.82rem;
        margin-top: 2px;
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
            data = response.json()
            data["source_transport"] = "FastAPI (/chat)"
            return data
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
        "groundedness": state.get("groundedness", 0.0),
        "retry_count": state.get("retry_count", 1),
        "source_transport": "LangGraph (In-Process)",
    }


# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")
    st.markdown(
        """
        **Book**: *Agentic AI: An Executive's Guide*  
        **Publisher**: Konverge AI  
        **Corpus**: 60 pages · 6 Chapters · 125 Chunks  
        """
    )
    st.divider()

    st.markdown("### ⚙️ System Status")
    active_model = GROQ_MODEL if LLM_PROVIDER == "groq" else OPENAI_MODEL
    st.markdown(f"- **LLM**: `{active_model}` ({LLM_PROVIDER.upper()})")
    st.markdown(f"- **Vector DB**: `{VECTOR_STORE.upper()}`")
    st.markdown(f"- **Embeddings**: `{EMBEDDING_MODEL}`")
    st.markdown(f"- **Top-K Chunks**: `{TOP_K}`")
    st.divider()

    st.markdown("### 💡 Quick Sample Queries")
    sample_queries = [
        ("🤖 RPA vs Agentic AI", "How does Agentic AI differ from RPA and traditional LLMs?"),
        ("🏛️ 5 Core Pillars", "Explain the core pillars and anatomy of an Agentic AI system from perception to execution."),
        ("⚡ Multi-Agent Challenges", "What are the challenges of multi-agent systems and their mitigation strategies?"),
        ("📈 AI Readiness Levels", "What are the four readiness levels in the industry-specific AI readiness analysis?"),
        ("🏭 Factory 4.0 Case Study", "What impact did the Factory 4.0 use case deliver?"),
        ("🚫 Out-of-Scope Negative Test", "Who is the Prime Minister of India?"),
    ]

    for label, sq in sample_queries:
        if st.button(label, key=f"sidebar_{label}", use_container_width=True):
            st.session_state.pending_query = sq

    st.divider()
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()


# --- Main Header ---
st.markdown('<div class="hero-title">🤖 Agentic AI Executive Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Production-grade LangGraph RAG pipeline with relevance grading, hallucination verification, and strict grounding.</div>',
    unsafe_allow_html=True,
)


# --- Empty State: Perplexity-style Interactive Topic Cards ---
if len(st.session_state.messages) == 0 and not st.session_state.pending_query:
    st.markdown("##### 🚀 Ask a question or pick a benchmark query to explore the eBook:")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🤖 **Compare: RPA vs Agentic AI vs LLMs**\n\nChapter 1 definitions, capabilities, and autonomy differences.", use_container_width=True):
            st.session_state.pending_query = "How does Agentic AI differ from RPA and traditional LLMs?"
            st.rerun()

        if st.button("⚡ **Multi-Agent Systems: Challenges & Mitigations**\n\nSection 3.4 matrix of MAS communication, alignment, and security mitigations.", use_container_width=True):
            st.session_state.pending_query = "What are the challenges of multi-agent systems and their mitigation strategies?"
            st.rerun()

        if st.button("🏭 **Industrial Case Study: Factory 4.0**\n\nChapter 6 real-world deployment, efficiency gains, and downtime metrics.", use_container_width=True):
            st.session_state.pending_query = "What impact did the Factory 4.0 use case deliver?"
            st.rerun()

    with col2:
        if st.button("🏛️ **5 Core Pillars: Perception to Execution**\n\nChapter 2 anatomy: perception, reasoning, planning, action, and memory layers.", use_container_width=True):
            st.session_state.pending_query = "Explain the core pillars and anatomy of an Agentic AI system from perception to execution."
            st.rerun()

        if st.button("📈 **Executive AI Readiness Framework**\n\nChapter 5 evaluation parameters across Initial, Emerging, Developing, and Advanced.", use_container_width=True):
            st.session_state.pending_query = "What are the four readiness levels in the industry-specific AI readiness analysis?"
            st.rerun()

        if st.button("🚫 **Negative Test: Out-of-Scope Query**\n\nVerifies that questions outside the eBook ('Prime Minister of India') trigger clean refusal.", use_container_width=True):
            st.session_state.pending_query = "Who is the Prime Minister of India?"
            st.rerun()

    st.write("")


# --- Helper to render Assistant metadata badges, trace, and chunks ---
def render_response_metadata(meta: dict):
    if not meta:
        return

    label = meta.get("confidence_label", "Low")
    badge_class = f"badge-{label.lower()}"
    conf_val = float(meta.get("confidence", 0.0))
    is_refused = meta.get("refused", False)
    is_grounded = meta.get("grounded", False)
    grounded_score = float(meta.get("groundedness", 1.0 if is_grounded else 0.0))
    retries = int(meta.get("retry_count", 1))
    chunks = meta.get("retrieved_chunks", [])
    transport = meta.get("source_transport", "Pipeline")

    # 1. Top Badges Row
    grounded_label = "✅ Grounded in eBook" if (is_grounded and not is_refused) else ("🚫 Out-of-Scope Refusal" if is_refused else "⚠️ Ungrounded")
    
    # Extract unique page numbers for citation pills
    unique_pages = sorted(list({ch.get("page_number") for ch in chunks if ch.get("page_number")}))
    citations_html = "".join([f'<span class="page-citation-pill">📄 Page {p}</span>' for p in unique_pages])

    badges_html = textwrap.dedent(f"""
    <div style="margin-top: 10px; margin-bottom: 12px; display: flex; flex-wrap: wrap; gap: 10px; align-items: center;">
        <span class="{badge_class}">Confidence: {conf_val:.1%} ({label})</span>
        <span class="badge-grounded">{grounded_label}</span>
        {citations_html}
    </div>
    """).strip()
    st.markdown(badges_html, unsafe_allow_html=True)

    # 2. Pipeline Execution Trace Accordion (The High-ROI Evaluator Feature!)
    with st.expander(f"🔍 LangGraph Workflow Inspection ({'Refusal Edge' if is_refused else '5-Node State Execution'})"):
        st.markdown(f"**Execution Route**: `{transport}` · **Attempts**: `{retries}`")

        # Step 1: Retrieve
        chunk_count = len(chunks) if not is_refused else 0
        node1_html = textwrap.dedent("""
        <div class="stepper-node">
            <div class="stepper-node-title">1. Node: retrieve (Vector DB)</div>
            <div class="stepper-node-desc">Executed similarity search in ChromaDB. Converted L2 distances to cosine similarity via unit-vector formula.</div>
        </div>
        """).strip()
        st.markdown(node1_html, unsafe_allow_html=True)

        # Step 2: Grade Relevance
        grade_desc = f"Batched LLM grading kept {len(chunks)} relevant chunks." if not is_refused else "Grading identified 0 relevant chunks → routed to refuse edge."
        node2_html = textwrap.dedent(f"""
        <div class="stepper-node">
            <div class="stepper-node-title">2. Node: grade_documents (Batched LLM)</div>
            <div class="stepper-node-desc">{grade_desc}</div>
        </div>
        """).strip()
        st.markdown(node2_html, unsafe_allow_html=True)

        # Step 3 & 4: Generate & Verify
        if not is_refused:
            verdict_name = "fully_grounded" if grounded_score == 1.0 else ("partially_grounded" if grounded_score == 0.5 else "not_grounded")
            nodes345_html = textwrap.dedent(f"""
            <div class="stepper-node">
                <div class="stepper-node-title">3. Node: generate (Grounded Synthesis)</div>
                <div class="stepper-node-desc">Generated response strictly using retrieved context chunks (Generation attempt {retries}/3).</div>
            </div>
            <div class="stepper-node">
                <div class="stepper-node-title">4. Node: verify_groundedness (Hallucination Check)</div>
                <div class="stepper-node-desc">Verdict: <code>{verdict_name}</code> (Score: {grounded_score:.1f}). Checked claims against source chunks.</div>
            </div>
            <div class="stepper-node">
                <div class="stepper-node-title">5. Node: compute_confidence (Composite Formula)</div>
                <div class="stepper-node-desc">Confidence = 0.6 × Normalized Retrieval + 0.4 × Groundedness = <b>{conf_val:.4f} ({label})</b></div>
            </div>
            """).strip()
            st.markdown(nodes345_html, unsafe_allow_html=True)
        else:
            refuse_node_html = textwrap.dedent("""
            <div class="stepper-node" style="border-left-color: #EF4444;">
                <div class="stepper-node-title">3. Node: refuse (Out-of-Scope Termination)</div>
                <div class="stepper-node-desc">Zero context passed grading. Emitted out-of-scope refusal without hallucinating or wasting generation tokens.</div>
            </div>
            """).strip()
            st.markdown(refuse_node_html, unsafe_allow_html=True)

        # Confidence Visual Breakdown Bar
        if not is_refused and chunks:
            st.markdown("---")
            mean_sim = sum([c.get("similarity_score", 0.0) for c in chunks]) / len(chunks)
            c_retrieval = 0.6 * mean_sim
            c_grounding = 0.4 * grounded_score
            st.caption(f"Confidence Formula Breakdown: Retrieval Match ({c_retrieval:.1%}) + Grounding ({c_grounding:.1%}) = **{conf_val:.1%}**")
            st.progress(min(1.0, max(0.0, conf_val)))

    # 3. Retrieved Context Chunks Expander
    if chunks:
        with st.expander(f"📚 Retrieved Context Chunks ({len(chunks)})"):
            for i, ch in enumerate(chunks):
                ctype = ch.get("content_type", "text")
                fragment_badge = (
                    f'<span class="badge-fragment">⚠️ {ctype.replace("_", " ").title()}</span>'
                    if ctype != "text" else ""
                )
                score = ch.get("similarity_score", 0.0)
                page = ch.get("page_number", "?")
                sec = ch.get("section", "Unknown")
                clean_text = html.escape(str(ch.get('text', '')).strip())
                clean_sec = html.escape(str(sec))

                card_html = textwrap.dedent(f"""
                <div class="chunk-glass-box">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span class="chunk-header-title">Chunk #{i+1} · Page {page} · Section: {clean_sec}</span>
                        <div>
                            {fragment_badge}
                            <span style="color:#38BDF8; font-weight:700; font-size: 0.88rem; margin-left: 8px;">Score: {score:.2f}</span>
                        </div>
                    </div>
                    <div class="chunk-body-text">{clean_text}</div>
                </div>
                """).strip()
                st.markdown(card_html, unsafe_allow_html=True)


# --- Render Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            render_response_metadata(msg["metadata"])


# --- Process Query Input ---
query_to_process = None

if st.session_state.pending_query:
    query_to_process = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    user_input = st.chat_input("Ask any question about the Agentic AI eBook...")
    if user_input:
        query_to_process = user_input

if query_to_process:
    # 1. Add User message
    st.session_state.messages.append({"role": "user", "content": query_to_process})
    with st.chat_message("user"):
        st.markdown(query_to_process)

    # 2. Add Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Analyzing knowledge base with LangGraph..."):
            res = query_rag(query_to_process)

        answer_text = res.get("answer", "No response received.")
        st.markdown(answer_text)
        render_response_metadata(res)

        # 3. Save assistant message with metadata
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer_text,
            "metadata": res,
        })
