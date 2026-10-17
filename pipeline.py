"""
pipeline.py — end-to-end orchestration.

    query -> Router -> [specialist agents] -> Verification Agent
          -> bounded revision -> Report Generator -> cited report

This is the single entry point used by the CLI, the FastAPI service and the
Streamlit UI, so all three demonstrate exactly the same behaviour.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List, Optional

import config  # noqa: F401  (loads .env before any backend is chosen)
from agents.analysts import FundamentalAgent, RiskAgent, SentimentAgent, TechnicalAgent
from agents.data_collector import DataCollectionAgent
from agents.orchestrator import Router
from agents.report_generator import render_markdown
from agents.synthesis import SynthesisAgent
from agents.validator import VerificationAgent
from core import Claim, Report
from data.corpus import load_corpus
from data.universe import COMPANIES, UnknownCompanyError, resolve_ticker, supported_tickers
from rag.store import EvidenceStore

log = logging.getLogger(__name__)

# Live data goes stale; the frozen snapshot never does. 0 disables expiry.
CACHE_TTL_S = float(os.environ.get("FINRESEARCH_CACHE_TTL_S", "3600"))

AGENT_CLASSES = {
    "fundamental": FundamentalAgent,
    "technical": TechnicalAgent,
    "sentiment": SentimentAgent,
    "risk": RiskAgent,
}


class ResearchPipeline:
    def __init__(self, prefer_live: bool = True, prefer_neural: bool = True,
                 threshold: float = 0.60, router: Optional[Router] = None,
                 analysts: str = "template", llm=None, synthesize: bool = True):
        """analysts: "template" (deterministic, default) | "llm" (LLM-written
        claims, verified) | "auto" (LLM if a backend is reachable, else template)."""
        self.prefer_live = prefer_live
        self.synthesize = synthesize            # add the "Overall view" section
        self.prefer_neural = prefer_neural
        self.threshold = threshold
        self.router = router or Router().fit()
        self._cache: Dict[str, dict] = {}
        self._lock = threading.RLock()          # the pipeline is shared across API threads
        self.llm = None
        if analysts in ("llm", "auto"):
            from llm import get_llm
            self.llm = llm or get_llm()
            if not self.llm.available:
                if analysts == "llm":
                    raise RuntimeError(
                        "analysts='llm' needs a backend: set GROQ_API_KEY or run Ollama")
                self.llm = None
        self.analysts = "llm" if self.llm else "template"

    @property
    def supported(self) -> List[str]:
        """Tickers this pipeline can analyse: all companies live, the 3 snapshot ones offline."""
        return supported_tickers(offline=not self.prefer_live)

    def _make_agent(self, name: str, store):
        if self.llm is not None:
            from agents.llm_analysts import LLMAnalyst
            return LLMAnalyst(store, name, self.llm)
        return AGENT_CLASSES[name](store)

    # ------------------------------------------------------------- context
    def _prepare(self, ticker: str) -> dict:
        with self._lock:
            ctx = self._cache.get(ticker)
            if ctx is not None:
                fresh = (ctx["mode"] != "live" or CACHE_TTL_S <= 0
                         or time.time() - ctx["built_at"] < CACHE_TTL_S)
                if fresh:
                    return ctx
                log.info("cache for %s expired; re-ingesting", ticker)
            ctx = self._build(ticker)
            self._cache[ticker] = ctx
            return ctx

    def _build(self, ticker: str) -> dict:
        t0 = time.perf_counter()
        chunks, prices, news_gold, mode = load_corpus(ticker, prefer_live=self.prefer_live)

        collector = DataCollectionAgent()
        indicators = collector.compute(ticker, prices)
        chunks = list(chunks) + collector.derived_evidence(ticker, indicators, prices)

        # Sentiment is scored once, at ingestion, and the aggregate is written
        # back as derived evidence so that sentiment claims are verifiable in
        # exactly the same way filing claims are.
        sentiment_detail = self._score_sentiment(ticker, chunks)
        if sentiment_detail.get("chunk") is not None:
            chunks.append(sentiment_detail.pop("chunk"))

        store = EvidenceStore().index(chunks)
        store.source_prices, store.source_ticker = prices, ticker
        ctx = {
            "store": store,
            "indicators": indicators,
            "prices": prices,
            "news_gold": news_gold,
            "sentiment_detail": {ticker: sentiment_detail},
            "mode": mode,
            "ingest_seconds": round(time.perf_counter() - t0, 3),
            "chunks": len(chunks),
            "built_at": time.time(),
        }
        log.info("ingested %s: mode=%s chunks=%d embedder=%s in %.2fs", ticker, mode,
                 len(chunks), store.embedder.name, ctx["ingest_seconds"])
        return ctx

    # ------------------------------------------------------- sentiment pass
    @staticmethod
    def _score_sentiment(ticker: str, chunks) -> dict:
        from core import Chunk
        from sentiment_model import get_scorer

        scorer = get_scorer()
        news = [c for c in chunks if c.ticker == ticker and c.doc_type == "news"]
        if not news:
            return {}
        per_chunk = {c.chunk_id: scorer.label(c.text) for c in news}
        labels = list(per_chunk.values())
        pos, neg, neu = (labels.count(x) for x in ("positive", "negative", "neutral"))
        total = len(labels)
        net = round((pos - neg) / total, 2)
        src = f"derived_{ticker}_sentiment"
        text = (
            f"Sentiment scoring record for {ticker} produced by {scorer.name} over "
            f"{total} indexed news items. Positive items: {pos}. Negative items: "
            f"{neg}. Neutral items: {neu}. Net sentiment score is {net}, computed as "
            f"(positive minus negative) divided by total items."
        )
        # doc_type is deliberately NOT "news": this record is derived analysis,
        # and letting it back into the news population would make the agent count
        # its own output as an input on the next pass.
        chunk = Chunk(chunk_id=Chunk.make_id(src, text), text=text, source=src,
                      doc_type="analysis", ticker=ticker, published="2026-08-31",
                      meta={"provenance": "computed", "model": scorer.name})
        return {"per_chunk": per_chunk, "model": scorer.name,
                "evidence_chunk_id": chunk.chunk_id, "chunk": chunk,
                "counts": {"positive": pos, "negative": neg, "neutral": neu},
                "net": net}

    # ----------------------------------------------------------------- run
    def run(self, query: str, ticker: Optional[str] = None,
            force_agents: Optional[List[str]] = None) -> Report:
        t_start = time.perf_counter()
        ticker = ticker or resolve_ticker(query)
        if ticker is None:
            raise UnknownCompanyError(
                "no supported company named in the query; supported: "
                + ", ".join(c.name for c in COMPANIES if c.ticker in self.supported))
        if ticker not in self.supported:
            raise UnknownCompanyError(
                f"{ticker} is not available in {'offline' if not self.prefer_live else 'live'} "
                f"mode; available: {', '.join(self.supported)}")
        ctx = self._prepare(ticker)
        store: EvidenceStore = ctx["store"]

        t_route = time.perf_counter()
        selected = force_agents or self.router.route(query)
        route_time = time.perf_counter() - t_route

        agents = {n: self._make_agent(n, store) for n in selected}

        t_agents = time.perf_counter()
        claims: List[Claim] = []
        for agent in agents.values():
            claims.extend(agent.run(ticker, ctx))
        if self.synthesize:
            # Overall view: reconciles the specialists' inputs, deterministic, and its
            # claims are verified like everyone else's. Listed first in the report.
            synth = SynthesisAgent(store, set(selected))
            agents["synthesis"] = synth
            claims = synth.run(ticker, ctx) + claims
        agent_time = time.perf_counter() - t_agents

        validator = VerificationAgent(store, threshold=self.threshold)
        t_verify = time.perf_counter()
        surfaced, verdicts, loop_stats = validator.verify_with_revision(
            claims, agents, ticker, ctx)
        verify_time = time.perf_counter() - t_verify

        sections: Dict[str, List[Claim]] = {}
        for c in surfaced:
            sections.setdefault(c.section, []).append(c)

        report = Report(
            query=query, ticker=ticker, sections=sections, verdicts=verdicts,
            agents_invoked=selected,
            timings={
                "route_s": round(route_time, 4),
                "agents_s": round(agent_time, 4),
                "verify_s": round(verify_time, 4),
                "total_s": round(time.perf_counter() - t_start, 4),
                "ingest_s": ctx["ingest_seconds"],
            },
            loop_stats=loop_stats, mode=ctx["mode"],
            validator_calls=validator.calls, store=store,
            withheld=list(validator.withheld), first_pass=list(validator.first_pass),
            analysts=self.analysts,
        )
        log.info("report ticker=%s agents=%s analysts=%s claims=%d surfaced=%d withheld=%d "
                 "total=%.2fs", ticker, ",".join(selected), self.analysts, len(verdicts),
                 report.surfaced_count, len(validator.withheld), report.timings["total_s"])
        return report

    def markdown(self, report: Report) -> str:
        return render_markdown(report, store=getattr(report, "store", None))


# ------------------------------------------------------------------- CLI
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Multi-Agent Financial Research Assistant")
    ap.add_argument("query", nargs="?",
                    default="Should I look at TCS for a 1-year hold?")
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--offline", action="store_true",
                    help="force the frozen snapshot corpus")
    ap.add_argument("--llm", action="store_true",
                    help="analysts write claims with an LLM (Groq or Ollama); "
                         "every claim is still verified")
    args = ap.parse_args()

    from logging_setup import setup_logging
    setup_logging()

    pipe = ResearchPipeline(prefer_live=not args.offline,
                            analysts="llm" if args.llm else "template")
    rep = pipe.run(args.query, ticker=args.ticker)
    print(pipe.markdown(rep))
    print()
    print("timings:", rep.timings, "| corpus:", rep.mode,
          "| validator calls:", rep.validator_calls)
