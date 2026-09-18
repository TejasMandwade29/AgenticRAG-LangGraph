"""
Ingestion pipeline: PDF → extract text → section-aware chunking → embeddings → vector store.

Usage:
    python -m src.ingest                 # Incremental (skip if index exists)
    python -m src.ingest --recreate      # Wipe and rebuild from scratch
"""

import re
import sys
import argparse
import statistics
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_STORE,
    CHROMA_PERSIST_DIR,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    get_chapter_for_page,
)


# ---------------------------------------------------------------------------
# 1. PDF Extraction
# ---------------------------------------------------------------------------

# Section header patterns found in the PDF.
# Matches numbered sections ("1.1", "2.3") and lettered subsections
# ("A. Decision Tree", "B. Industry-Wise Readiness", "C. AI Readiness Analysis").
SECTION_HEADER_RE = re.compile(
    r"^(?:(\d+\.\d+)|([A-D])\.{1,2})\s+(.+)",
    re.MULTILINE,
)

# Chapter title headers (e.g., "INTRODUCTION TO AGENTIC AI")
CHAPTER_TITLE_RE = re.compile(
    r"^(INTRODUCTION|ANATOMY|MULTI-AGENT|ORCHESTRATING|YOUR READINESS|PRACTICAL)",
    re.MULTILINE,
)

# Pages whose content is a visual diagram (branch relationships lost in extraction)
DIAGRAM_PAGES: set[int] = {48, 49}

# Pages whose content is a multi-column table with ambiguous row-column mapping
TABLE_FRAGMENT_PAGES: set[int] = {50}


def extract_pages(pdf_path: Path) -> list[dict]:
    """Extract text from each page of the PDF with page numbers."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if len(text) < 15:
            continue  # Skip empty / image-only pages

        # Clean up common PDF artifacts
        text = _clean_text(text)
        pages.append({
            "page_number": i + 1,  # 1-indexed
            "text": text,
        })
    return pages


def _clean_text(text: str) -> str:
    """Clean extracted PDF text of common artifacts."""
    # Fix smart quotes encoded oddly
    text = text.replace("ΓÇÖ", "'").replace("ΓÇ£", '"').replace("ΓÇ¥", '"')
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    
    # Fix bullet character artifacts
    text = text.replace("∩┐╜", "•")
    
    # Remove page footer "AGENTIC AI FOR EXECUTIVES"
    text = re.sub(r"\n?AGENTIC AI FOR EXECUTIVES\s*$", "", text.strip())
    
    # Remove lone page numbers at the end
    text = re.sub(r"\n\d{1,2}\s*$", "", text.strip())
    
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Section-Aware Chunking
# ---------------------------------------------------------------------------

def detect_section_title(text: str) -> str | None:
    """Try to detect a section header at the start of the text."""
    match = SECTION_HEADER_RE.search(text[:200])
    if match:
        # group(1) = numbered prefix (e.g. "3.4"), group(2) = letter prefix (e.g. "A")
        # group(3) = the rest of the title
        prefix = match.group(1) or match.group(2)
        title = match.group(3).strip()
        return f"{prefix}. {title}" if match.group(2) else f"{prefix} {title}"
    return None


def chunk_pages(pages: list[dict]) -> list[Document]:
    """
    Section-aware chunking strategy:
    1. Group text by detected sections within each page
    2. Use RecursiveCharacterTextSplitter with section boundaries as preferred split points
    3. Attach rich metadata to each chunk
    """
    # Build section-aware separators: prefer splitting on section headers, then paragraphs, then sentences
    separators = [
        "\n\n",               # Paragraph breaks (strongest boundary)
        "\n",                 # Line breaks
        ". ",                 # Sentence ends
        ", ",                 # Clause breaks
        " ",                  # Word breaks (last resort)
    ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    documents = []
    current_section = "Introduction"

    for page_data in pages:
        page_num = page_data["page_number"]
        text = page_data["text"]

        # Detect section in this page
        section = detect_section_title(text)
        if section:
            current_section = section

        # Get chapter info
        chapter_num, chapter_title = get_chapter_for_page(page_num)

        # Determine content_type for this page
        if page_num in DIAGRAM_PAGES:
            content_type = "diagram_fragment"
        elif page_num in TABLE_FRAGMENT_PAGES:
            content_type = "table_fragment"
        else:
            content_type = "text"

        # Split the page text
        chunks = splitter.split_text(text)

        for chunk in chunks:
            # Check if this chunk starts a new section
            chunk_section = detect_section_title(chunk)
            if chunk_section:
                current_section = chunk_section

            doc = Document(
                page_content=chunk,
                metadata={
                    "page_number": page_num,
                    "chapter": chapter_num,
                    "chapter_title": chapter_title or "N/A",
                    "section_title": current_section,
                    "source": "Ebook-Agentic-AI.pdf",
                    "content_type": content_type,
                },
            )
            documents.append(doc)

    return documents


# ---------------------------------------------------------------------------
# 3. Embedding & Vector Store
# ---------------------------------------------------------------------------

def get_embedding_function():
    """Return the configured embedding function."""
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


def store_in_chroma(documents: list[Document], recreate: bool = False):
    """Store documents in ChromaDB."""
    from langchain_chroma import Chroma

    embedding_fn = get_embedding_function()

    if recreate:
        import shutil
        if Path(CHROMA_PERSIST_DIR).exists():
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print(f"  Deleted existing ChromaDB at {CHROMA_PERSIST_DIR}")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embedding_fn,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="agentic_ai_ebook",
    )
    print(f"  Stored {len(documents)} chunks in ChromaDB at {CHROMA_PERSIST_DIR}")
    return vectorstore


def store_in_pinecone(documents: list[Document], recreate: bool = False):
    """Store documents in Pinecone."""
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    if not PINECONE_API_KEY:
        print("ERROR: PINECONE_API_KEY not set. Use VECTOR_STORE=chroma for local testing.")
        sys.exit(1)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]

    if recreate and PINECONE_INDEX_NAME in existing_indexes:
        pc.delete_index(PINECONE_INDEX_NAME)
        print(f"  Deleted existing Pinecone index '{PINECONE_INDEX_NAME}'")
        existing_indexes.remove(PINECONE_INDEX_NAME)

    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"  Created Pinecone index '{PINECONE_INDEX_NAME}' (dim={EMBEDDING_DIMENSION})")

    embedding_fn = get_embedding_function()
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embedding_fn,
        index_name=PINECONE_INDEX_NAME,
    )
    print(f"  Stored {len(documents)} chunks in Pinecone index '{PINECONE_INDEX_NAME}'")
    return vectorstore


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def print_chunk_samples(documents: list[Document], n: int = 5):
    """Print a few sample chunks for inspection."""
    print(f"\n{'='*80}")
    print(f"SAMPLE CHUNKS (showing {n} of {len(documents)})")
    print(f"{'='*80}")

    # Show diverse samples: first, middle, and specifically table-heavy chunks
    indices = [0, len(documents) // 4, len(documents) // 2, 3 * len(documents) // 4, len(documents) - 1]
    indices = sorted(set(i for i in indices if 0 <= i < len(documents)))[:n]

    for idx in indices:
        doc = documents[idx]
        meta = doc.metadata
        print(f"\n--- Chunk {idx} | Page {meta['page_number']} | Ch.{meta['chapter']} | "
              f"Section: {meta['section_title']} | {len(doc.page_content)} chars ---")
        # Show first 500 chars of the chunk
        preview = doc.page_content[:500]
        if len(doc.page_content) > 500:
            preview += "..."
        print(preview)

    # Also find and show a table-heavy chunk
    print(f"\n--- TABLE-HEAVY CHUNKS ---")
    table_keywords = ["Capability", "Description", "Example", "Challenge", "Mitigation", "Parameter",
                      "Evaluation", "Readiness", "Level", "Aspect", "LLMs", "Agents"]
    table_chunks = [
        (i, d) for i, d in enumerate(documents)
        if any(kw in d.page_content for kw in table_keywords) and
        d.page_content.count("\n") > 8  # Tables have many short lines
    ]
    for idx, doc in table_chunks[:3]:
        meta = doc.metadata
        print(f"\n--- Chunk {idx} | Page {meta['page_number']} | Ch.{meta['chapter']} | "
              f"Section: {meta['section_title']} | {len(doc.page_content)} chars ---")
        preview = doc.page_content[:600]
        if len(doc.page_content) > 600:
            preview += "..."
        print(preview)


def print_ingestion_summary(documents: list[Document]):
    """Print summary statistics about the ingestion."""
    sizes = [len(d.page_content) for d in documents]
    pages_covered = sorted(set(d.metadata["page_number"] for d in documents))
    chapters_covered = sorted(set(d.metadata["chapter"] for d in documents if d.metadata["chapter"]))

    print(f"\n{'='*80}")
    print(f"INGESTION SUMMARY")
    print(f"{'='*80}")
    print(f"  Total chunks:        {len(documents)}")
    print(f"  Avg chunk size:      {statistics.mean(sizes):.0f} chars")
    print(f"  Min chunk size:      {min(sizes)} chars")
    print(f"  Max chunk size:      {max(sizes)} chars")
    print(f"  Median chunk size:   {statistics.median(sizes):.0f} chars")
    print(f"  Pages covered:       {len(pages_covered)} (pages {pages_covered[0]}-{pages_covered[-1]})")
    print(f"  Chapters covered:    {chapters_covered}")
    print(f"  Chunk size config:   {CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap")
    print(f"  Vector store:        {VECTOR_STORE}")
    print(f"  Embedding model:     {EMBEDDING_MODEL} (dim={EMBEDDING_DIMENSION})")


def main():
    parser = argparse.ArgumentParser(description="Ingest PDF into vector store")
    parser.add_argument("--recreate", action="store_true", help="Wipe and rebuild the index")
    parser.add_argument("--dry-run", action="store_true", help="Extract and chunk only, don't store")
    args = parser.parse_args()

    print(f"PDF path: {PDF_PATH}")
    if not PDF_PATH.exists():
        print(f"ERROR: PDF not found at {PDF_PATH}")
        sys.exit(1)

    # Step 1: Extract
    print("\n[1/3] Extracting text from PDF...")
    pages = extract_pages(PDF_PATH)
    print(f"  Extracted {len(pages)} pages with text")

    # Step 2: Chunk
    print("\n[2/3] Chunking with section-aware splitting...")
    documents = chunk_pages(pages)
    print(f"  Created {len(documents)} chunks")

    # Print samples and summary
    print_chunk_samples(documents)
    print_ingestion_summary(documents)

    if args.dry_run:
        print("\n[DRY RUN] Skipping vector store storage.")
        return

    # Step 3: Store
    print(f"\n[3/3] Storing in {VECTOR_STORE}...")
    if VECTOR_STORE == "chroma":
        store_in_chroma(documents, recreate=args.recreate)
    elif VECTOR_STORE == "pinecone":
        store_in_pinecone(documents, recreate=args.recreate)
    else:
        print(f"ERROR: Unknown VECTOR_STORE '{VECTOR_STORE}'. Use 'chroma' or 'pinecone'.")
        sys.exit(1)

    print("\nIngestion complete!")


if __name__ == "__main__":
    main()
