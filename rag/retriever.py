"""
TF-IDF & Cosine Similarity Retriever for Campus RAG
===================================================
"""

import re
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

from rag.ingest import ingest_documents
from rag.embeddings import EmbeddingModel


CAMPUS_SYNONYMS = {
    "services": "services facilities support amenities welfare centres",
    "service": "services facilities support amenities welfare centres",
    "opening": "open opening hours timings schedule operational",
    "closing": "close closing hours timings schedule",
    "hours": "hours timings timing schedule time open close",
    "timing": "timings timing hours schedule open close",
    "timings": "timings timing hours schedule open close",
    "curfew": "curfew timings gates entry night out permission",
    "rules": "rules regulations code policy guidelines",
    "amenities": "amenities facilities rooms beds ac furniture accommodation",
    "facility": "facilities amenities rooms services centres",
    "facilities": "facilities amenities rooms services centres",
    "hostel": "hostel residential rooms mess accommodation curfew warden",
    "attendance": "attendance percentage 75 classes requirement condonation",
    "exam": "examinations exam cat fat assessment test",
    "exams": "examinations exam cat fat assessment test",
    "doctor": "medical healthcare doctor nurse clinic hospital emergency",
    "medical": "medical centre healthcare doctor first aid emergency ambulance",
    "wifi": "wifi internet network connection it helpdesk",
    "mess": "mess food dining breakfast lunch dinner meal",
    "sports": "sports football cricket basketball badminton gymnasium",
    "book": "books borrow borrowing library card circulation renewal",
    "fine": "fine fines penalty charges late return",
    "transport": "transportation bus shuttle routes transit pass"
}


def expand_campus_query(query: str) -> str:
    """
    Expands a user query with domain-relevant synonyms to bridge vocabulary gaps.
    """
    clean_words = re.findall(r"\b[a-zA-Z]+\b", query.lower())
    expanded_terms = list(clean_words)
    
    for word in clean_words:
        if word in CAMPUS_SYNONYMS:
            expanded_terms.append(CAMPUS_SYNONYMS[word])
            
    return " ".join(expanded_terms)


class Retriever:
    """
    RAG Retriever for campus knowledge documents.
    """

    def __init__(self):
        self.documents = ingest_documents()
        self.embedding_model = EmbeddingModel()
        self.embeddings = None

        if self.documents:
            texts = [doc["text"] for doc in self.documents]
            self.embedding_model.fit(texts)
            self.embeddings = self.embedding_model.encode(texts)

    def _retrieve_single(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.03
    ) -> List[Dict[str, Any]]:
        """Retrieves passages for a single sub-query with domain intent boosting."""
        if not self.documents or self.embeddings is None:
            return []

        expanded_q = expand_campus_query(query)
        query_vector = self.embedding_model.encode([expanded_q])
        similarities = cosine_similarity(query_vector, self.embeddings)[0]

        q_lower = query.lower()
        # Apply domain-level scoring adjustments
        for idx, doc in enumerate(self.documents):
            src = doc["source"]
            txt = doc["text"]
            boost = 1.0

            # Domain keyword match
            if "hostel" in q_lower and src == "hostel.txt":
                boost += 0.35
            if "library" in q_lower and src == "library.txt":
                boost += 0.35
            if any(w in q_lower for w in ["campus service", "student service", "services available", "services on campus", "medical", "bus", "sports", "wifi"]) and src == "campus_services.txt":
                boost += 0.35
            if any(w in q_lower for w in ["academic", "attendance", "cat", "fat", "grade", "gpa", "exam"]) and src == "academic.txt":
                boost += 0.35
            
            # Overview boost for broad queries
            if any(w in q_lower for w in ["overview", "what are the", "facilities available", "services available", "what student services", "what services"]) and "0. Overview" in txt:
                boost += 0.25

            similarities[idx] *= boost

        ranked_indices = similarities.argsort()[::-1]

        results = []
        for idx in ranked_indices:
            score = float(similarities[idx])
            if score < min_score and len(results) >= 1:
                continue

            doc = self.documents[idx]
            results.append({
                "source": doc["source"],
                "title": doc.get("title", doc["source"]),
                "chunk_id": doc["chunk_id"],
                "text": doc["text"],
                "score": round(score, 4)
            })

            if len(results) >= top_k:
                break

        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.03
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top-k ranked document chunks, supporting compound multi-intent queries.
        """
        if not self.documents or self.embeddings is None:
            return []

        # Check for compound query joined by conjunctions
        q_lower = query.lower()
        sub_queries = [s.strip() for s in re.split(r"\band\b|&|\bas well as\b|\balso\b", q_lower) if len(s.strip().split()) >= 2]

        if len(sub_queries) > 1:
            combined_results = []
            seen_chunks = set()
            for sub_q in sub_queries:
                sub_res = self._retrieve_single(sub_q, top_k=2, min_score=min_score)
                for r in sub_res:
                    key = (r["source"], r["chunk_id"])
                    if key not in seen_chunks:
                        seen_chunks.add(key)
                        combined_results.append(r)
            if combined_results:
                return combined_results[:max(top_k, 4)]

        return self._retrieve_single(query, top_k=top_k, min_score=min_score)

    def available_sources(self) -> List[str]:
        return sorted(list(set(doc["source"] for doc in self.documents)))