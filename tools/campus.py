"""
Campus Information Tool
=======================
Answers queries regarding campus facilities, libraries, hostels, academics, and student services
using Grounded Retrieval-Augmented Generation (RAG).
"""

from typing import Dict, Any
from tools.rag import retrieve


def campus_lookup(query: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Campus lookup tool leveraging local campus knowledge base documents:
    - library.txt
    - hostel.txt
    - academic.txt
    - campus_services.txt
    """
    rag_result = retrieve(query=query, top_k=top_k)
    return {
        "success": True,
        "tool": "campus",
        "query": query,
        "knowledge_base": [
            "library.txt",
            "hostel.txt",
            "academic.txt",
            "campus_services.txt"
        ],
        "result_count": rag_result["result_count"],
        "results": rag_result["results"]
    }