"""
rag/embedder.py — pluggable embedding backend.

Design note (this matters for the viva): the verification mechanism must not
depend on any one embedding model or any paid API. The system therefore defines
an Embedder interface with two interchangeable implementations:

  * SentenceTransformerEmbedder — all-MiniLM-L6-v2, used when the model weights
    are available locally. This is the production path.
  * TfidfSvdEmbedder — a deterministic character+word n-gram TF-IDF projected to
    a dense vector with truncated SVD. Zero download, zero network, and good
    enough for numeric/entity overlap on short financial claims. This is the
    offline fallback and the one used for reproducible evaluation runs.

Both return L2-normalised dense vectors, so cosine similarity is a dot product
and the downstream validator is identical either way.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


class Embedder:
    dim: int
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


def _l2(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SentenceTransformerEmbedder(Embedder):
    name = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: WPS433
        self._m = SentenceTransformer(model_name)
        get_dim = getattr(self._m, "get_embedding_dimension", None) \
            or self._m.get_sentence_embedding_dimension
        self.dim = get_dim()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return _l2(np.asarray(self._m.encode(list(texts), show_progress_bar=False), dtype=np.float32))


class TfidfSvdEmbedder(Embedder):
    """Deterministic offline embedder. Must be fitted on the corpus first."""

    name = "tfidf-svd-256"

    def __init__(self, dim: int = 256):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline

        self.dim = dim
        self._word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=1,
        )
        self._char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1,
        )
        self._svd = TruncatedSVD(n_components=dim, random_state=42)
        self._fitted = False
        self._make_pipeline = make_pipeline

    def fit(self, corpus: Sequence[str]) -> "TfidfSvdEmbedder":
        from scipy.sparse import hstack

        w = self._word.fit_transform(corpus)
        c = self._char.fit_transform(corpus)
        x = hstack([w, c]).tocsr()
        n_comp = min(self.dim, max(2, min(x.shape) - 1))
        self._svd.n_components = n_comp
        self._svd.fit(x)
        self.dim = n_comp
        self._fitted = True
        return self

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from scipy.sparse import hstack

        if not self._fitted:
            raise RuntimeError("TfidfSvdEmbedder.fit(corpus) must be called first")
        w = self._word.transform(list(texts))
        c = self._char.transform(list(texts))
        x = hstack([w, c]).tocsr()
        return _l2(np.asarray(self._svd.transform(x), dtype=np.float32))


_NEURAL: Optional[Embedder] = None


def _neural_embedder() -> Embedder:
    """Load the sentence-transformer once per process, not once per store."""
    global _NEURAL
    if _NEURAL is None:
        _NEURAL = SentenceTransformerEmbedder()
    return _NEURAL


def build_embedder(corpus: Sequence[str], prefer_neural: bool = True) -> Embedder:
    """Return the embedder for this run.

    FINRESEARCH_EMBEDDER selects explicitly: "tfidf" (deterministic, offline —
    what the reproducible evaluation and the unit tests use), "neural"
    (fail loudly if unavailable) or "auto" (default: neural if it loads, else
    tfidf). The chosen backend is reported on the embedder's `name`.
    """
    mode = os.environ.get("FINRESEARCH_EMBEDDER", "auto").lower()
    if mode == "tfidf":
        return TfidfSvdEmbedder().fit(corpus)
    if mode == "neural":
        return _neural_embedder()
    if prefer_neural:
        try:
            return _neural_embedder()
        except Exception as exc:
            log.warning("neural embedder unavailable (%s); using tfidf-svd", exc)
    return TfidfSvdEmbedder().fit(corpus)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity for L2-normalised vectors."""
    return a @ b.T
