"""
Campus RAG Tool Wrapper
=======================
"""

from typing import Dict, Any, List
from rag.retriever import Retriever

_retriever = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def retrieve(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Retrieves relevant document passages for campus inquiries."""
    retriever = get_retriever()
    results = retriever.retrieve(query=query, top_k=top_k)
    return {
        "success": True,
        "tool": "campus_rag",
        "query": query,
        "result_count": len(results),
        "results": results
    }


def available_documents() -> List[str]:
    retriever = get_retriever()
    return retriever.available_sources()