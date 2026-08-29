"""
Vectorization and Embedding for Campus RAG
===========================================
"""

from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingModel:
    """
    TF-IDF Vectorizer with sublinear term frequency scaling.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode"
        )
        self.fitted = False

    def fit(self, documents):
        if not documents:
            return
        self.vectorizer.fit(documents)
        self.fitted = True

    def encode(self, documents):
        if not self.fitted:
            raise RuntimeError("Embedding model has not been fitted.")
        return self.vectorizer.transform(documents)