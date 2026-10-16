"""agents/base.py — the contract every analysis agent obeys."""

from __future__ import annotations

from typing import List, Optional

from core import Claim
from rag.store import EvidenceStore


class Agent:
    name: str = "agent"
    section: str = "General"

    def __init__(self, store: EvidenceStore):
        self.store = store

    # ------------------------------------------------------------------
    def cite(self, evidence_query: str, ticker: str, k: int = 3) -> List[str]:
        """Retrieve the evidence this agent is about to speak from.

        An agent is not permitted to emit a claim without first retrieving. The
        ids returned here are the citation it must attach.
        """
        hits = self.store.query(evidence_query, k=k, ticker=ticker)
        return [c.chunk_id for c, _ in hits]

    def claim(self, text: str, cited: List[str]) -> Claim:
        return Claim.new(agent=self.name, section=self.section,
                         text=text, cited_chunk_ids=cited)

    # ------------------------------------------------------------------
    def run(self, ticker: str, context: dict) -> List[Claim]:
        raise NotImplementedError

    def revise(self, claim: Claim, feedback: str, ticker: str,
               context: dict) -> Optional[Claim]:
        """Bounded revision: re-ground the claim against actual evidence.

        The default strategy is the conservative one — re-retrieve, and rewrite
        the claim so that it asserts only what the retrieved evidence states. An
        agent that cannot do this returns None, and the claim is dropped rather
        than shown with a fabricated number.
        """
        return None
