"""
All prompt templates used by the LangGraph pipeline nodes.

Keeping prompts in a dedicated module makes them easy to iterate on
without touching node logic, and easy to review for correctness.
"""

from langchain_core.prompts import PromptTemplate


# ---------------------------------------------------------------------------
# 1. Grading prompt (BATCHED — single call for all retrieved chunks)
# ---------------------------------------------------------------------------
# Fix: one LLM call returns a JSON array of relevant indices,
# replacing the previous 1-call-per-chunk approach (5 calls → 1).
#
# The LLM receives all chunks numbered, and replies with a JSON object
# containing a list of the indices that are relevant to the question.
# ---------------------------------------------------------------------------

GRADE_PROMPT_TEMPLATE = """\
You are a relevance grader for a RAG system grounded in the book \
"Agentic AI: An Executive's Guide" by Konverge AI.

Given a user question and a numbered list of retrieved document chunks, \
identify which chunks contain information that is relevant to answering \
the question. A chunk is relevant if it contains facts, definitions, \
comparisons, examples, or frameworks that directly help answer the question.

Respond with ONLY a valid JSON object in this exact format:
{{"relevant_indices": [<list of 0-based integer indices>]}}

If no chunks are relevant, respond with:
{{"relevant_indices": []}}

Do not include any explanation. Do not include markdown. Output JSON only.

User question: {question}

Retrieved chunks:
{numbered_chunks}
"""

GRADE_PROMPT = PromptTemplate(
    input_variables=["question", "numbered_chunks"],
    template=GRADE_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# 2. Generation prompt
# ---------------------------------------------------------------------------
# {diagram_disclaimer} is injected conditionally:
#   - Empty string when all retrieved chunks have content_type == "text"
#   - The full disclaimer block when any chunk is a diagram_fragment
#     or table_fragment
# ---------------------------------------------------------------------------

DIAGRAM_DISCLAIMER = """\
IMPORTANT — DIAGRAM/TABLE CONTENT WARNING:
Some of the retrieved context below was extracted from a visual diagram \
or multi-column table in the original PDF. The spatial relationships \
(which branch connects to which label, or which cell value belongs to \
which column) may not be fully recoverable from the extracted text.
When using this content:
- State at a high level what the diagram or table covers.
- Present the individual elements (labels, values, stage names) that \
ARE clearly and explicitly stated in the text.
- Do NOT invent or assert relationships between elements that are not \
explicitly stated. If a specific mapping is ambiguous, say so rather \
than guessing.
"""

GENERATE_PROMPT_TEMPLATE = """\
You are an expert AI assistant answering questions strictly based on the \
provided context from the book "Agentic AI: An Executive's Guide" \
by Konverge AI and Emergence AI.

{diagram_disclaimer}
Rules:
1. Answer ONLY using information present in the provided context chunks below.
2. Do not use any external knowledge, training data, or assumptions beyond \
what the context states.
3. If the context does not contain enough information to answer the question \
fully, say so explicitly — do not fabricate.
4. Be specific: cite section numbers (e.g. "Section 3.4"), statistics, \
named frameworks, or examples from the text wherever possible.
5. Structure your answer clearly. Use short paragraphs or bullet points \
where appropriate.

Context:
{formatted_context}

Question: {question}

Answer:"""

GENERATE_PROMPT = PromptTemplate(
    input_variables=["question", "formatted_context", "diagram_disclaimer"],
    template=GENERATE_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# 3. Groundedness verification prompt
# ---------------------------------------------------------------------------
# Returns one of three verdicts: fully_grounded / partially_grounded /
# not_grounded. The LLM must output only the JSON object.
# ---------------------------------------------------------------------------

VERIFY_PROMPT_TEMPLATE = """\
You are a groundedness verifier for a RAG system. Your job is to check \
whether a generated answer is fully supported by the provided source chunks.

Definitions:
- "fully_grounded": every factual claim in the answer is directly supported \
by the source chunks.
- "partially_grounded": the answer mixes supported claims with at least one \
claim that is not found in the source chunks (i.e. it adds knowledge \
beyond what the sources state).
- "not_grounded": the answer contains claims that directly contradict the \
sources, or the answer is largely fabricated with no support in the chunks.

Respond with ONLY a valid JSON object in this exact format:
{{"verdict": "<fully_grounded|partially_grounded|not_grounded>"}}

Do not include any explanation. Output JSON only.

Source chunks:
{formatted_context}

Generated answer:
{answer}
"""

VERIFY_PROMPT = PromptTemplate(
    input_variables=["formatted_context", "answer"],
    template=VERIFY_PROMPT_TEMPLATE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_context(documents) -> str:
    """
    Format a list of LangChain Documents into a numbered context block
    for injection into generation and verification prompts.
    """
    parts = []
    for i, doc in enumerate(documents):
        meta = doc.metadata
        header = (
            f"[Chunk {i + 1} | Page {meta.get('page_number', '?')} | "
            f"Chapter {meta.get('chapter', '?')} | "
            f"Section: {meta.get('section_title', 'Unknown')}]"
        )
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(parts)


def format_numbered_chunks(documents) -> str:
    """
    Format documents as a numbered list for the batched grading prompt.
    Zero-based indices match the JSON array the grader returns.
    """
    parts = []
    for i, doc in enumerate(documents):
        meta = doc.metadata
        label = (
            f"[{i}] Page {meta.get('page_number', '?')} | "
            f"Section: {meta.get('section_title', 'Unknown')}"
        )
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(parts)


def get_diagram_disclaimer(documents) -> str:
    """
    Return the disclaimer block if any chunk is a diagram or table fragment;
    return an empty string otherwise.
    """
    flagged_types = {"diagram_fragment", "table_fragment"}
    for doc in documents:
        if doc.metadata.get("content_type", "text") in flagged_types:
            return DIAGRAM_DISCLAIMER
    return ""
