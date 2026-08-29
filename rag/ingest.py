"""
Document Ingestion & Chunking for Campus RAG
============================================
Loads knowledge base files from the data/ directory and chunks them for semantic retrieval.
"""

from pathlib import Path
from typing import List, Dict, Any

# Resolve path relative to this script so it works from any execution context
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_documents() -> List[Dict[str, str]]:
    """
    Loads all .txt files in the data directory.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    documents = []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        try:
            text = file_path.read_text(encoding="utf-8").strip()
            if text:
                documents.append({
                    "source": file_path.name,
                    "title": file_path.stem.replace("_", " ").title(),
                    "text": text
                })
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")

    return documents


def chunk_document(
    doc: Dict[str, str],
    chunk_size: int = 100,
    overlap: int = 15
) -> List[str]:
    """
    Chunks document text preserving section integrity and adding document context.
    """
    text = doc["text"]
    title = doc.get("title", "Campus Document")
    sections = [s.strip() for s in text.split("\n\n") if s.strip()]
    chunks = []

    for section in sections:
        words = section.split()
        
        # Skip isolated title-only headers like "LIBRARY INFORMATION"
        if len(words) <= 3 and any(h in section.upper() for h in ["INFORMATION", "GUIDE", "MANUAL", "RULES"]):
            continue

        # Prefix with document title context if not already prefixed
        contextual_text = f"[{title}] {section}" if not section.startswith(f"[{title}]") else section

        if len(words) <= chunk_size:
            chunks.append(contextual_text)
        else:
            # Word-based sliding window for oversized paragraphs
            for i in range(0, len(words), chunk_size - overlap):
                window_words = words[i:i + chunk_size]
                chunk_str = f"[{title}] " + " ".join(window_words).strip()
                if chunk_str:
                    chunks.append(chunk_str)
                if i + chunk_size >= len(words):
                    break

    return chunks


def ingest_documents() -> List[Dict[str, Any]]:
    """
    Ingests and chunks all campus knowledge base documents with metadata.
    """
    raw_docs = load_documents()
    chunks = []

    for doc in raw_docs:
        doc_chunks = chunk_document(doc)
        for idx, chunk_text_content in enumerate(doc_chunks, start=1):
            chunks.append({
                "source": doc["source"],
                "title": doc["title"],
                "chunk_id": idx,
                "text": chunk_text_content
            })

    return chunks


if __name__ == "__main__":
    c = ingest_documents()
    print(f"Ingested {len(c)} chunks from {DATA_DIR}")