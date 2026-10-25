"""api/main.py — FastAPI service.  Run: uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health            liveness + configuration (no auth)
  POST /research          run a report and wait for it (504 after the request timeout)
  POST /jobs              start a report in the background -> 202 {job_id}
  GET  /jobs/{job_id}     poll: queued | running | done | failed
  POST /verify            run the Verification Agent on one claim
  POST /audit             check every sentence of a pasted text against the data

Configuration (environment or .env — see .env.example)
  FINRESEARCH_API_KEY          if set, every endpoint except /health needs `X-API-Key`
  FINRESEARCH_RATE_LIMIT       requests per minute per client (default 30, 0 = off)
  FINRESEARCH_MAX_CONCURRENT   reports computed at the same time (default 2)
  FINRESEARCH_REQUEST_TIMEOUT_S  sync /research timeout (default 120)
  FINRESEARCH_ANALYSTS         template (default) | llm | auto
  FINRESEARCH_OFFLINE          1 = frozen snapshot corpus only, never touch the network
  FINRESEARCH_CORS             comma-separated allowed browser origins

The rate limiter and job store live in process memory: run one worker, or put a
shared store (Redis) behind them before scaling out — see DEPLOY.md.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Deque, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import config  # noqa: E402,F401  (loads .env)
from data.corpus import DataUnavailableError  # noqa: E402
from data.universe import COMPANIES, UnknownCompanyError, find_tickers, supported_tickers  # noqa: E402
from logging_setup import setup_logging  # noqa: E402
from pipeline import ResearchPipeline  # noqa: E402

setup_logging()
log = logging.getLogger("api")

OFFLINE = os.environ.get("FINRESEARCH_OFFLINE", "") in ("1", "true", "yes")
SUPPORTED = supported_tickers(offline=OFFLINE)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


API_KEY = os.environ.get("FINRESEARCH_API_KEY", "")
RATE_LIMIT = _env_int("FINRESEARCH_RATE_LIMIT", 30)
MAX_CONCURRENT = max(1, _env_int("FINRESEARCH_MAX_CONCURRENT", 2))
REQUEST_TIMEOUT_S = _env_int("FINRESEARCH_REQUEST_TIMEOUT_S", 120)
JOB_TTL_S = 3600
MAX_JOBS = 200

app = FastAPI(title="Multi-Agent Financial Research Assistant", version="1.0.0")
_origins = [o.strip() for o in os.environ.get(
    "FINRESEARCH_CORS", "http://localhost:8501,http://127.0.0.1:8501").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"],
                   allow_headers=["*"])

_pipe_lock = threading.Lock()
_pipe: Optional[ResearchPipeline] = None
_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT, thread_name_prefix="research")
_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT)


def get_pipeline() -> ResearchPipeline:
    """Built on first use, so importing the module (tests, docs) stays cheap."""
    global _pipe
    with _pipe_lock:
        if _pipe is None:
            _pipe = ResearchPipeline(
                prefer_live=not OFFLINE,
                analysts=os.environ.get("FINRESEARCH_ANALYSTS", "template"))
        return _pipe


# ------------------------------------------------------------------ security
def require_api_key(request: Request) -> None:
    if not API_KEY:
        return
    supplied = request.headers.get("x-api-key", "")
    if not secrets.compare_digest(supplied.encode(), API_KEY.encode()):
        raise HTTPException(401, "missing or invalid X-API-Key")


_hits: Dict[str, Deque[float]] = {}
_hits_lock = threading.Lock()


def rate_limit(request: Request) -> None:
    """Sliding one-minute window per client address (in-memory, single process)."""
    if RATE_LIMIT <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _hits_lock:
        q = _hits.setdefault(client, deque())
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            retry = max(1, int(60 - (now - q[0])))
            raise HTTPException(429, f"rate limit of {RATE_LIMIT}/minute exceeded",
                                headers={"Retry-After": str(retry)})
        q.append(now)


GUARDS = [Depends(require_api_key), Depends(rate_limit)]


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    log.info("%s %s -> %d in %.0fms rid=%s", request.method, request.url.path,
             response.status_code, (time.perf_counter() - t0) * 1000, rid)
    return response


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse({"detail": "internal error"}, status_code=500)


# -------------------------------------------------------------------- models
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


class AuditRequest(BaseModel):
    text: str = Field(min_length=10, max_length=8000)
    ticker: str | None = Field(default=None, description="Company the text is about; "
                               "default: the company it names most often.")


def _check_ticker(ticker: str | None) -> None:
    if ticker is not None and ticker not in SUPPORTED:
        raise HTTPException(422, f"unsupported ticker {ticker!r}; supported: {SUPPORTED}")


def _check_query(q: "Query") -> None:
    """Reject early (422) a request that names no company we can analyse."""
    _check_ticker(q.ticker)
    if q.ticker is None:
        found = find_tickers(q.query)
        if not found or found[0] not in SUPPORTED:
            names = ", ".join(c.name for c in COMPANIES if c.ticker in SUPPORTED)
            raise HTTPException(422, f"no supported company named in the query; supported: {names}")


def _payload(rep, pipe: ResearchPipeline) -> dict:
    withheld_text = {w.claim_id: c.text for c, w in rep.withheld}
    return {
        "query": rep.query,
        "ticker": rep.ticker,
        "agents_invoked": rep.agents_invoked,
        "analysts": rep.analysts,
        "grounding_rate": rep.grounding_rate,
        "corpus_mode": rep.mode,
        "timings": rep.timings,
        "sections": {
            sec: [{"text": c.text,
                   "confidence": rep.verdicts[c.claim_id].confidence,
                   "numeric_support": rep.verdicts[c.claim_id].numeric_support,
                   "sources": c.cited_chunk_ids}
                  for c in claims]
            for sec, claims in rep.sections.items()
        },
        "withheld": [{**v.to_dict(), "text": withheld_text.get(v.claim_id)}
                     for v in rep.verdicts.values() if v.status == "flagged"],
        "markdown": pipe.markdown(rep),
    }


def _run_report(q: Query) -> dict:
    pipe = get_pipeline()
    with _semaphore:                       # bound concurrent heavy work
        return _payload(pipe.run(q.query, ticker=q.ticker), pipe)


# ----------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    pipe = get_pipeline()
    return {"status": "ok", "version": app.version, "analysts": pipe.analysts,
            "llm": pipe.llm.name if pipe.llm else None, "auth_required": bool(API_KEY),
            "offline": OFFLINE, "supported_tickers": SUPPORTED,
            "companies": {c.ticker: c.name for c in COMPANIES if c.ticker in SUPPORTED}}


@app.post("/research", dependencies=GUARDS)
async def research(q: Query):
    _check_query(q)
    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(_executor, _run_report, q),
                                      timeout=REQUEST_TIMEOUT_S)
    except DataUnavailableError as exc:
        raise HTTPException(503, str(exc)) from None
    except asyncio.TimeoutError:
        raise HTTPException(504, f"report not finished within {REQUEST_TIMEOUT_S}s; use "
                                 "POST /jobs for long (LLM) reports") from None


# ---- background jobs
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _purge_jobs() -> None:
    cutoff = time.time() - JOB_TTL_S
    for jid in [j for j, v in _jobs.items() if v["updated"] < cutoff]:
        del _jobs[jid]


def _job_worker(job_id: str, q: Query) -> None:
    def update(**kw):
        with _jobs_lock:
            _jobs[job_id].update(kw, updated=time.time())

    update(status="running")
    try:
        update(status="done", result=_run_report(q))
    except DataUnavailableError as exc:     # an expected, user-explainable failure
        update(status="failed", error=str(exc))
    except Exception as exc:                # noqa: BLE001 — reported to the poller
        log.exception("job %s failed", job_id)
        update(status="failed", error=type(exc).__name__)


@app.post("/jobs", status_code=202, dependencies=GUARDS)
def start_job(q: Query):
    _check_query(q)
    with _jobs_lock:
        _purge_jobs()
        if len(_jobs) >= MAX_JOBS:
            raise HTTPException(503, "too many stored jobs; retry later")
        job_id = uuid.uuid4().hex
        _jobs[job_id] = {"status": "queued", "created": time.time(), "updated": time.time()}
    _executor.submit(_job_worker, job_id, q)
    return {"job_id": job_id, "status": "queued", "poll": f"/jobs/{job_id}"}


@app.get("/jobs/{job_id}", dependencies=GUARDS)
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        snapshot = dict(job) if job else None
    if snapshot is None:
        raise HTTPException(404, "unknown or expired job")
    return {"job_id": job_id, "status": snapshot["status"],
            "result": snapshot.get("result"), "error": snapshot.get("error")}


@app.post("/verify", dependencies=GUARDS)
def verify_one(req: VerifyRequest):
    """Expose the Verification Agent on its own.

    The caller supplies the citation (or none): the endpoint never invents one,
    so the confidence reflects what the caller actually asserted."""
    from agents.validator import VerificationAgent
    from core import Claim

    _check_ticker(req.ticker)
    store = get_pipeline()._prepare(req.ticker)["store"]
    claim = Claim.new(agent="external", section="ad-hoc", text=req.text,
                      cited_chunk_ids=req.cited_chunk_ids or [])
    return VerificationAgent(store).verify(claim).to_dict()


@app.post("/audit", dependencies=GUARDS)
async def audit(req: AuditRequest):
    """Audit someone else's text: every sentence is verified, contradicted,
    unsupported or uncheckable against the company's live data."""
    from agents.auditor import audit_text

    _check_ticker(req.ticker)

    def run() -> dict:
        pipe = get_pipeline()
        with _semaphore:
            return audit_text(req.text, pipe, ticker=req.ticker).to_dict()

    loop = asyncio.get_running_loop()
    try:
        return await asyncio.wait_for(loop.run_in_executor(_executor, run),
                                      timeout=REQUEST_TIMEOUT_S)
    except (UnknownCompanyError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    except DataUnavailableError as exc:
        raise HTTPException(503, str(exc)) from None
    except asyncio.TimeoutError:
        raise HTTPException(504, f"audit not finished within {REQUEST_TIMEOUT_S}s") from None
