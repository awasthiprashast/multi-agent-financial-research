"""
rag/store.py — the evidence store.

Every document the Data Collection Agent pulls is chunked, embedded and indexed
here with its source and publication date. Two things retrieve from it:

  1. the analysis agents, when they need evidence to make a claim;
  2. the Verification Agent, when it independently re-retrieves evidence for a
     claim that has already been made.

(2) is the important one. The validator never trusts the citation an agent
attached — it re-retrieves from scratch and checks whether the cited chunk comes
back. That is what makes the citation falsifiable rather than decorative.

ChromaDB is used when available; a numpy-backed index with identical semantics
is used otherwise, so evaluation runs are reproducible with no service running.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from core import Chunk
from rag.embedder import Embedder, build_embedder

log = logging.getLogger(__name__)


class EvidenceStore:
    def __init__(self, embedder: Optional[Embedder] = None, use_chroma: bool = True):
        self.chunks: Dict[str, Chunk] = {}
        self._ids: List[str] = []
        self._mat: Optional[np.ndarray] = None
        self.embedder = embedder
        self._use_chroma = use_chroma
        self._collection = None
        # Raw price series this store's computed evidence was derived from. The
        # Verification Agent recomputes indicators from it (see validator).
        self.source_prices: Optional[dict] = None
        self.source_ticker: Optional[str] = None
        self._recompute_cache: Optional[list] = None
        self._recompute_ind: Optional[dict] = None

    # ---------------------------------------------------------------- index
    def index(self, chunks: Sequence[Chunk]) -> "EvidenceStore":
        for c in chunks:
            self.chunks[c.chunk_id] = c
        self._ids = list(self.chunks.keys())
        corpus = [self.chunks[i].text for i in self._ids]

        if self.embedder is None:
            self.embedder = build_embedder(corpus)

        self._mat = self.embedder.encode(corpus)

        if self._use_chroma:
            try:
                import chromadb

                # EphemeralClient instances share one in-process backend, so a
                # fixed collection name lets one ticker's store overwrite
                # another's. Every store gets its own collection.
                client = chromadb.EphemeralClient()
                # HNSW is approximate and its default search width (ef_search=10)
                # silently returns fewer results than asked for. The evidence set
                # here is hundreds of chunks at most, so make the search wide enough
                # to be exact; ranking is then deterministic (see _top_k).
                self._collection = client.create_collection(
                    name=f"evidence_{uuid.uuid4().hex[:12]}",
                    configuration={"hnsw": {"space": "cosine", "ef_construction": 400,
                                            "ef_search": 400, "max_neighbors": 48}},
                )
                self._collection.add(
                    ids=self._ids,
                    embeddings=[v.tolist() for v in self._mat],
                    documents=corpus,
                    metadatas=[
                        {
                            "source": self.chunks[i].source,
                            "doc_type": self.chunks[i].doc_type,
                            "ticker": self.chunks[i].ticker,
                            "published": self.chunks[i].published,
                        }
                        for i in self._ids
                    ],
                )
            except Exception as exc:
                log.warning("ChromaDB unavailable (%s); using numpy-flat index", exc)
                self._collection = None
        return self

    # ------------------------------------------------------------- retrieve
    def query(self, text: str, k: int = 5,
              ticker: Optional[str] = None) -> List[Tuple[Chunk, float]]:
        if self._mat is None or self.embedder is None:
            raise RuntimeError("EvidenceStore.index() must be called first")

        qv = self.embedder.encode([text])[0]

        if self._collection is not None:
            where = {"ticker": ticker} if ticker else None
            try:
                # Over-fetch, then break ties ourselves: identical similarities are
                # ordered arbitrarily by the ANN index, and which chunk survives the
                # cut would otherwise differ from process to process.
                res = self._collection.query(
                    query_embeddings=[qv.tolist()],
                    n_results=min(max(4 * k, 64), len(self._ids)),
                    where=where,
                )
                cands = [(self.chunks[cid], float(1.0 - dist))
                         for cid, dist in zip(res["ids"][0], res["distances"][0])]
                return self._top_k(cands, k)
            except Exception:
                pass  # fall through to numpy path

        sims = self._mat @ qv
        cands = [(self.chunks[cid], float(sims[i])) for i, cid in enumerate(self._ids)
                 if not ticker or self.chunks[cid].ticker == ticker]
        return self._top_k(cands, k)

    @staticmethod
    def _top_k(cands: List[Tuple[Chunk, float]], k: int) -> List[Tuple[Chunk, float]]:
        """Highest similarity first; exact ties broken by chunk id (deterministic)."""
        cands.sort(key=lambda cs: (-round(cs[1], 6), cs[0].chunk_id))
        return cands[:k]

    def get(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunks.get(chunk_id)

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def backend(self) -> str:
        return "chromadb" if self._collection is not None else "numpy-flat"
