# Deploying

This is a research prototype hardened for a small deployment (one host, a handful of
users). Read **Limits** before putting it in front of the public.

## 1. Configure

```bash
cp .env.example .env        # Windows cmd: copy .env.example .env
```

Fill in only what you need. For anything reachable from a network set at least:

| Variable | Why |
|---|---|
| `FINRESEARCH_API_KEY` | every endpoint except `/health` then requires an `X-API-Key` header |
| `FINRESEARCH_CORS` | the exact browser origins allowed to call the API |
| `FINRESEARCH_RATE_LIMIT` | requests per minute per client (default 30) |
| `GROQ_API_KEY` | only if you want LLM analysts |

Generate an API key: `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
Secrets live in `.env` or in your platform's secret store — never in the image, the
repository, or a chat window.

## 2. Run

**Docker (recommended)**

```bash
docker compose up --build                 # API on http://localhost:8000
docker compose --profile ui up --build    # + Streamlit UI on http://localhost:8501
docker compose --profile local-llm up     # + a local Ollama server
```

The default image is small and uses TF-IDF embeddings and the lexicon sentiment scorer.
For FinBERT and MiniLM build with `--build-arg WITH_NEURAL=1` (adds PyTorch, ~2.5 GB;
the first start downloads the model weights into `/home/app/.cache`, so mount a
volume there if you want them to persist).

**Without Docker**

```bash
pip install -e ".[api,llm,ui]"            # add ,neural for FinBERT + MiniLM
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
streamlit run frontend/streamlit_app.py
```

## 3. Use the API

```bash
curl -s localhost:8000/health

# quick report (template analysts answer in well under a second)
curl -s -X POST localhost:8000/research -H "X-API-Key: $KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "Should I look at TCS for a 1-year hold?"}'

# LLM analysts take 30 s – 5 min: start a job and poll it
curl -s -X POST localhost:8000/jobs -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
     -d '{"query": "Complete analysis of Infosys"}'
curl -s localhost:8000/jobs/<job_id> -H "X-API-Key: $KEY"
```

Interactive docs: `/docs`. Every response carries an `X-Request-ID`; the same id is in
the server log line. Set `FINRESEARCH_LOG_JSON=1` for JSON logs (the image does).

## 4. Operate

- **Health:** `GET /health` (used by the Docker healthcheck).
- **Logs:** one line per request and one per report (`ticker`, agents, claims surfaced
  and withheld, timings). Unhandled errors are logged with a traceback and returned to
  the client as a bare `500`.
- **Cache:** live data per ticker is refreshed after `FINRESEARCH_CACHE_TTL_S` (default
  3600 s). The on-disk cache (`data/cache`) is a Docker volume; deleting it only costs
  re-downloads.
- **Backups:** nothing here is stateful. Jobs and rate-limit counters are in memory.

## Limits — what this is not yet

- **Single process.** The rate limiter and the job store are in memory, so run one
  worker. Scaling out needs Redis (or similar) behind them.
- **Unofficial data sources.** Prices and fundamentals come from Yahoo Finance via
  `yfinance`, headlines from Google News RSS. Both have terms of use and no uptime
  guarantee; check them before commercial or public use, or swap in a licensed feed.
- **Coverage.** Three companies are supported end to end (TCS, Reliance, Infosys);
  see `TICKER_ALIASES` and `ENTITY_FORMS` to add more. Live fundamentals are four
  headline ratios, not filings.
- **Verification is bounded.** Numbers and a defined set of directional statements are
  checked against data. Interpretation and speculation are not (see the README).
- **Not investment advice.** The disclaimer in every report is not optional. If this
  is offered to the public, have it reviewed against the rules that apply to you
  (for India, SEBI's research-analyst and investment-adviser regulations).
- **The container image has not been built or run by the author's tooling** in the
  environment this file was written in (Docker was unavailable). CI builds it and
  hits `/health` on every push; check that job before relying on the image.
