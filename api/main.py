"""api/main.py — FastAPI service.  Run: uvicorn api.main:app --reload

Environment:
  FINRESEARCH_ANALYSTS   template (default) | llm | auto
  FINRESEARCH_CORS       comma-separated allowed origins (default: localhost only)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException      # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field           # noqa: E402

import config                                   # noqa: E402,F401  (loads .env)
from agents.orchestrator import TICKER_ALIASES  # noqa: E402
from pipeline import ResearchPipeline           # noqa: E402

SUPPORTED = sorted(set(TICKER_ALIASES.values()))

app = FastAPI(title="Multi-Agent Financial Research Assistant", version="1.0.0")
_origins = [o.strip() for o in os.environ.get(
    "FINRESEARCH_CORS", "http://localhost:8501,http://127.0.0.1:8501").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"],
                   allow_headers=["*"])

_pipe = ResearchPipeline(analysts=os.environ.get("FINRESEARCH_ANALYSTS", "template"))


class Query(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    ticker: str | None = None


class VerifyRequest(BaseModel):
    text: str = Field(min_length=3, max_length=1000)
    ticker: str = "TCS.NS"
    cited_chunk_ids: list[str] | None = Field(
        default=None,
        description="Chunks the claim asserts as its source. Omit to verify with no "
                    "citation (citation_consistent will then be false).")


def _check_ticker(ticker: str | None) -> None:
    if ticker is not None and ticker not in SUPPORTED:
        raise HTTPException(422, f"unsupported ticker {ticker!r}; supported: {SUPPORTED}")


@app.get("/health")
def health():
    return {"status": "ok", "version": app.version, "analysts": _pipe.analysts,
            "supported_tickers": SUPPORTED}


@app.post("/research")
def research(q: Query):
    _check_ticker(q.ticker)
    rep = _pipe.run(q.query, ticker=q.ticker)
    return {
        "query": rep.query,
        "ticker": rep.ticker,
        "agents_invoked": rep.agents_invoked,
        "analysts": rep.analysts,
        "grounding_rate": rep.grounding_rate,
        "corpus_mode": rep.mode,
        "timings": rep.timings,
        "sections": {
            sec: [
                {
                    "text": c.text,
                    "confidence": rep.verdicts[c.claim_id].confidence,
                    "numeric_support": rep.verdicts[c.claim_id].numeric_support,
                    "sources": c.cited_chunk_ids,
                }
                for c in claims
            ]
            for sec, claims in rep.sections.items()
        },
        "withheld": [{**v.to_dict(), "text": next(
            (c.text for c, w in rep.withheld if w.claim_id == v.claim_id), None)}
            for v in rep.verdicts.values() if v.status == "flagged"],
        "markdown": _pipe.markdown(rep),
    }


@app.post("/verify")
def verify_one(req: VerifyRequest):
    """Expose the Verification Agent on its own.

    The caller supplies the citation (or none): the endpoint never invents one,
    so the confidence reflects what the caller actually asserted."""
    from agents.validator import VerificationAgent
    from core import Claim

    _check_ticker(req.ticker)
    store = _pipe._prepare(req.ticker)["store"]
    claim = Claim.new(agent="external", section="ad-hoc", text=req.text,
                      cited_chunk_ids=req.cited_chunk_ids or [])
    return VerificationAgent(store).verify(claim).to_dict()
