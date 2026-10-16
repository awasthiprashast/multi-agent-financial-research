"""
agents/llm_analysts.py — analysts that write their claims with a language model.

The template analysts in analysts.py copy numbers out of retrieved text, so they
cannot hallucinate and say little about the verifier. These agents are the real
test: an LLM reads retrieved evidence and writes free-form research sentences,
and every sentence goes through the same Verification Agent. Claims that fail
are sent back here (`revise`) with the verifier's reason, at most R times.

The LLM is told to copy numbers verbatim and never to compute new ones. It does
not always comply — measuring how often it does not is what eval/llm_eval.py is
for.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from agents.base import Agent
from core import Claim
from llm import LLM, LLMUnavailable

log = logging.getLogger(__name__)

SECTION_NAMES = {
    "fundamental": "Fundamental Analysis",
    "technical": "Technical Analysis",
    "sentiment": "Sentiment Analysis",
    "risk": "Risk Assessment",
}

EVIDENCE_QUERIES = {
    "fundamental": [
        "headline fundamentals price-to-earnings return on equity profit margin",
        "consolidated revenue from operations growth over previous financial year",
        "operating margin for the year bps movement versus prior year",
        "return on equity and debt to equity ratio balance sheet",
        "basic earnings per share and dividend yield",
    ],
    "technical": [
        "computed trend indicators RSI MACD moving average",
        "market data closing price highest lowest trading sessions",
    ],
    "sentiment": [
        "sentiment scoring record positive negative neutral net sentiment score",
        "news shares analysts brokerages announced",
    ],
    "risk": [
        "computed risk statistics volatility drawdown value at risk",
    ],
}

FOCUS = {
    "fundamental": "revenue, margins, returns, leverage, earnings per share and dividend",
    "technical": "price versus moving averages, RSI, MACD, Bollinger bands and period return",
    "sentiment": "the tone of recent news and what the sentiment scores show",
    "risk": "volatility, maximum drawdown and Value at Risk",
}

SYSTEM = (
    "You are an equity research analyst writing for an investor. Write ONLY from "
    "the numbered evidence you are given. Rules: (1) every number you write must be "
    "copied exactly from the evidence; never compute, round or estimate a number; "
    "(2) do not mention facts that are not in the evidence; (3) each claim is one "
    "sentence; (4) name the company; (5) answer with JSON only."
)


def _format_evidence(chunks) -> str:
    return "\n".join(f"[{i}] ({c.source}) {c.text}" for i, c in enumerate(chunks, 1))


def _parse_json(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


class LLMAnalyst(Agent):
    """One specialist (fundamental / technical / sentiment / risk) backed by an LLM."""

    def __init__(self, store, kind: str, llm: LLM, max_evidence: int = 6):
        super().__init__(store)
        self.name = kind
        self.section = SECTION_NAMES[kind]
        self.kind = kind
        self.llm = llm
        self.max_evidence = max_evidence
        self.llm_calls = 0
        self._evidence: Dict[str, list] = {}

    # ---------------------------------------------------------------- evidence
    def _gather(self, ticker: str) -> list:
        seen, chunks = set(), []
        per_query = 2 if self.kind == "fundamental" else 1
        for q in EVIDENCE_QUERIES[self.kind]:
            for chunk, _ in self.store.query(q, k=per_query, ticker=ticker):
                if chunk.chunk_id not in seen:
                    seen.add(chunk.chunk_id)
                    chunks.append(chunk)
        return chunks[: self.max_evidence]

    def _ask(self, user: str):
        self.llm_calls += 1
        raw = self.llm.generate(SYSTEM, user, json_mode=True)
        return _parse_json(raw)

    # --------------------------------------------------------------------- run
    def run(self, ticker: str, context: dict) -> List[Claim]:
        chunks = self._gather(ticker)
        if not chunks:
            return []
        self._evidence[ticker] = chunks
        user = (
            f"Company: {ticker}\nSection: {self.section}\n"
            f"Focus: {FOCUS[self.kind]}\n\nEvidence:\n{_format_evidence(chunks)}\n\n"
            "Write 3 to 5 research claims. Return JSON exactly like "
            '{"claims":[{"text":"...","sources":[1,2]}]} where sources are the '
            "evidence numbers each claim relies on."
        )
        try:
            data = self._ask(user)
        except LLMUnavailable:
            raise
        except Exception as exc:
            log.warning("%s analyst: LLM call failed (%s)", self.kind, exc)
            return []
        claims: List[Claim] = []
        for item in (data or {}).get("claims", [])[:6]:
            claim = self._to_claim(item, chunks)
            if claim is not None:
                claims.append(claim)
        return claims

    def _to_claim(self, item, chunks) -> Optional[Claim]:
        if isinstance(item, str):
            item = {"text": item, "sources": []}
        if not isinstance(item, dict):
            return None
        text = str(item.get("text", "")).strip()
        if len(text) < 15:
            return None
        cited = []
        for s in item.get("sources", []) or []:
            try:
                idx = int(s) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(chunks):
                cited.append(chunks[idx].chunk_id)
        return self.claim(text, cited)

    # ------------------------------------------------------------------ revise
    def revise(self, claim: Claim, feedback: str, ticker: str,
               context: dict) -> Optional[Claim]:
        chunks = self._evidence.get(ticker) or self._gather(ticker)
        user = (
            f"Company: {ticker}\n\nEvidence:\n{_format_evidence(chunks)}\n\n"
            f"Your earlier claim was rejected by a fact-checker.\n"
            f"Claim: {claim.text}\nProblem: {feedback}\n\n"
            "Rewrite it so that every number is copied exactly from the evidence and "
            "nothing unsupported remains; if it cannot be supported, restate only what "
            'the evidence says. Return JSON exactly like {"text":"...","sources":[1]}.'
        )
        try:
            data = self._ask(user)
        except Exception as exc:
            log.warning("%s analyst: revise failed (%s)", self.kind, exc)
            return None
        if not isinstance(data, dict):
            return None
        revised = self._to_claim(data, chunks)
        if revised is not None:
            revised.revision = claim.revision + 1
        return revised
