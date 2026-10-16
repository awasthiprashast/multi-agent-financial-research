"""
sentiment_model.py — financial sentiment scoring.

Primary: FinBERT (ProsusAI/finbert) run locally through transformers. No API, no
key, no per-call cost — the model is downloaded once and then runs offline.

Fallback: a financial-domain lexicon scorer (Loughran-McDonald style polarity
lists with negation handling). This exists so that every part of the system
remains runnable on a machine that cannot download model weights, and so that
evaluation runs are reproducible. Both expose the same `label()` interface, so
swapping them changes nothing downstream.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List

log = logging.getLogger(__name__)


class Scorer:
    name = "scorer"

    def label(self, text: str) -> str:
        raise NotImplementedError

    def labels(self, texts: List[str]) -> List[str]:
        return [self.label(t) for t in texts]


class FinBertScorer(Scorer):
    name = "FinBERT (ProsusAI/finbert)"

    def __init__(self):
        from transformers import pipeline  # noqa: WPS433

        self._pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert",
                              truncation=True, max_length=512)

    def label(self, text: str) -> str:
        return self._pipe(text[:1500])[0]["label"].lower()


# --------------------------------------------------------------------------
POSITIVE = {
    "advanced", "gain", "gains", "rose", "surge", "surged", "beat", "stronger",
    "strong", "growth", "improving", "improved", "resilience", "resilient",
    "healthy", "raised", "upgrade", "upgraded", "outperform", "record",
    "highest", "expansion", "robust", "positive", "recovery", "biggest",
}
NEGATIVE = {
    "slipped", "fell", "decline", "declined", "weak", "weaker", "softer",
    "concerns", "concern", "pressure", "downgrade", "downgraded", "cut",
    "trimming", "trimmed", "slowing", "loss", "losses", "risk", "warned",
    "cautioned", "miss", "missed", "underperform", "lawsuit", "probe",
    "extended", "worse",
}
NEGATORS = {"not", "no", "never", "without", "despite", "offset"}


class LexiconScorer(Scorer):
    name = "financial lexicon (offline fallback)"

    _TOKEN = re.compile(r"[a-z']+")

    def label(self, text: str) -> str:
        toks = self._TOKEN.findall(text.lower())
        score = 0
        for i, t in enumerate(toks):
            window = set(toks[max(0, i - 3):i])
            flip = -1 if window & NEGATORS else 1
            if t in POSITIVE:
                score += 1 * flip
            elif t in NEGATIVE:
                score -= 1 * flip
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"


_CACHED: Scorer | None = None


def get_scorer(prefer_finbert: bool = True) -> Scorer:
    """FINRESEARCH_SCORER = "lexicon" | "finbert" | "auto" (default).

    "finbert" fails loudly if the model cannot load; "auto" falls back to the
    lexicon scorer and logs the reason. The scorer is cached per process.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    mode = os.environ.get("FINRESEARCH_SCORER", "auto").lower()
    if mode == "finbert":
        _CACHED = FinBertScorer()
        return _CACHED
    if prefer_finbert and mode != "lexicon":
        try:
            _CACHED = FinBertScorer()
            return _CACHED
        except Exception as exc:
            log.warning("FinBERT unavailable (%s); using lexicon scorer", exc)
    _CACHED = LexiconScorer()
    return _CACHED
