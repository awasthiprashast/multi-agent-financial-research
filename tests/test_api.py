"""API tests with FastAPI's TestClient (offline snapshot corpus, template analysts)."""

import importlib
import time

import pytest
from fastapi.testclient import TestClient


def _client(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    import api.main as m
    m = importlib.reload(m)                       # re-read env-derived settings
    m._hits.clear()
    return m, TestClient(m.app)


def test_health_is_open_and_reports_config(monkeypatch):
    m, c = _client(monkeypatch, FINRESEARCH_API_KEY="secret")
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["auth_required"] is True and "TCS.NS" in body["supported_tickers"]
    assert "x-request-id" in r.headers


def test_research_requires_the_api_key_when_configured(monkeypatch):
    m, c = _client(monkeypatch, FINRESEARCH_API_KEY="secret")
    q = {"query": "How volatile is Reliance?"}
    assert c.post("/research", json=q).status_code == 401
    assert c.post("/research", json=q, headers={"X-API-Key": "wrong"}).status_code == 401
    ok = c.post("/research", json=q, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200
    body = ok.json()
    assert body["ticker"] == "RELIANCE.NS" and body["grounding_rate"] == 1.0
    assert "not investment advice" in body["markdown"].lower()


def test_no_key_configured_means_open(monkeypatch):
    monkeypatch.delenv("FINRESEARCH_API_KEY", raising=False)
    m, c = _client(monkeypatch)
    assert c.post("/research", json={"query": "How volatile is Reliance?"}).status_code == 200


def test_input_validation(monkeypatch):
    m, c = _client(monkeypatch)
    assert c.post("/research", json={"query": "x"}).status_code == 422             # too short
    assert c.post("/research", json={"query": "y" * 600}).status_code == 422        # too long
    r = c.post("/research", json={"query": "analyse it", "ticker": "NOPE.NS"})
    assert r.status_code == 422 and "unsupported ticker" in r.json()["detail"]
    assert c.post("/verify", json={"ticker": "TCS.NS"}).status_code == 422           # no text


def test_rate_limit_returns_429_with_retry_after(monkeypatch):
    m, c = _client(monkeypatch, FINRESEARCH_RATE_LIMIT=3)
    for _ in range(3):
        assert c.post("/verify", json={"text": "Return on equity was 51.2%."}).status_code == 200
    r = c.post("/verify", json={"text": "Return on equity was 51.2%."})
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_verify_does_not_invent_a_citation(monkeypatch):
    m, c = _client(monkeypatch, FINRESEARCH_RATE_LIMIT=0)
    r = c.post("/verify", json={"text": "Return on equity for the year was 51.2%.",
                                "ticker": "TCS.NS"}).json()
    assert r["citation_consistent"] is False          # nothing was cited, nothing assumed
    bad = c.post("/verify", json={"text": "Return on equity for the year was 74.9%."}).json()
    assert bad["status"] == "flagged" and bad["unsupported_numbers"]


def test_background_job_lifecycle(monkeypatch):
    m, c = _client(monkeypatch, FINRESEARCH_RATE_LIMIT=0)
    started = c.post("/jobs", json={"query": "Complete analysis of Infosys"})
    assert started.status_code == 202
    jid = started.json()["job_id"]
    for _ in range(100):
        job = c.get(f"/jobs/{jid}").json()
        if job["status"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "done", job
    assert job["result"]["ticker"] == "INFY.NS" and job["result"]["sections"]
    assert c.get("/jobs/does-not-exist").status_code == 404


def test_unhandled_error_is_hidden_from_the_client(monkeypatch):
    m, c = _client(monkeypatch, FINRESEARCH_RATE_LIMIT=0)
    monkeypatch.setattr(m, "_run_report", lambda q: (_ for _ in ()).throw(RuntimeError("secret detail")))
    r = TestClient(m.app, raise_server_exceptions=False).post("/research", json={"query": "full report on tcs"})
    assert r.status_code == 500 and "secret" not in r.text


def test_slow_report_times_out_with_504(monkeypatch):
    m, c = _client(monkeypatch, FINRESEARCH_RATE_LIMIT=0, FINRESEARCH_REQUEST_TIMEOUT_S=1)
    monkeypatch.setattr(m, "_run_report", lambda q: time.sleep(2.5))
    r = c.post("/research", json={"query": "full report on tcs"})
    assert r.status_code == 504 and "/jobs" in r.json()["detail"]


@pytest.fixture(autouse=True)
def _restore_api_module():
    yield
    import api.main as m
    importlib.reload(m)


def test_audit_endpoint_classifies_sentences(monkeypatch):
    m, c = _client(monkeypatch)
    text = ("Operating margin for the year was 24.3%, a movement of -70 bps against the prior year. "
            "TCS shares will reach Rs. 5,000 by FY27.")
    r = c.post("/audit", json={"text": text, "ticker": "TCS.NS"})
    assert r.status_code == 200
    body = r.json()
    assert [s["status"] for s in body["sentences"]] == ["verified", "uncheckable"]
    assert body["counts"]["verified"] == 1 and body["accuracy"] == 1.0
    assert c.post("/audit", json={"text": "Margins were 24.3% last year."}).status_code == 422
    assert c.post("/audit", json={"text": text, "ticker": "NOPE.NS"}).status_code == 422
