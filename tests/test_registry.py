"""Companies outside the curated 12: lookup and registration (no network)."""

import pytest

from agents.validator import VerificationAgent
from data import universe as U


@pytest.fixture
def clean_registry():
    before = (list(U.COMPANIES), dict(U.BY_TICKER), dict(U.NAME), dict(U.TICKER_ALIASES),
              list(U._PATTERNS))
    yield
    U.COMPANIES[:] = before[0]
    for target, saved in ((U.BY_TICKER, before[1]), (U.NAME, before[2]),
                          (U.TICKER_ALIASES, before[3])):
        target.clear()
        target.update(saved)
    U._PATTERNS[:] = before[4]
    U.search_companies.cache_clear()


def test_clean_name_and_aliases():
    assert U._clean_name("ASIAN PAINTS LTD.") == "Asian Paints"
    assert U._clean_name("Bajaj Finance Limited") == "Bajaj Finance"
    assert U._aliases_for("PAYTM.NS", "One97 Communications Limited") == ("one97 communications", "paytm")
    assert "bank" not in U._aliases_for("XBANK.NS", "X Bank Limited")      # too generic alone


def test_registering_a_company_makes_it_known_everywhere(clean_registry):
    co = U.register_company("ASIANPAINT.NS", "ASIAN PAINTS LTD.")
    assert U.register_company("ASIANPAINT.NS", "anything") is co          # idempotent
    assert "ASIANPAINT.NS" in U.supported_tickers(offline=False)
    assert "ASIANPAINT.NS" not in U.supported_tickers(offline=True)
    assert U.NAME["ASIANPAINT.NS"] == "Asian Paints"
    assert U.find_tickers("Asian Paints beat estimates while Reliance fell") == ["ASIANPAINT.NS", "RELIANCE.NS"]
    assert U.mentions("Asian Paints shares rise", "ASIANPAINT.NS")
    assert not U.mentions("Asian markets rise", "ASIANPAINT.NS")
    assert "ASIANPAINT.NS" in VerificationAgent.entity_forms()


def test_search_keeps_main_listings_and_dedupes_exchanges(monkeypatch, clean_registry):
    quotes = [
        {"symbol": "ASIANPAINT.BO", "quoteType": "EQUITY", "longname": "Asian Paints Limited"},
        {"symbol": "ASIANPAINT-BL.NS", "quoteType": "EQUITY", "longname": "Asian Paints Limited"},
        {"symbol": "ASIANPAINT.NS", "quoteType": "EQUITY", "longname": "Asian Paints Limited"},
        {"symbol": "ASIANPAINT", "quoteType": "ETF", "longname": "Not equity"},
    ]

    class Resp:
        def raise_for_status(self): ...
        def json(self): return {"quotes": quotes}

    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp())
    found = U.search_companies("asian paints")
    assert [(c.ticker, c.exchange) for c in found] == [("ASIANPAINT.NS", "NSE")]


def test_search_failure_returns_nothing_instead_of_raising(monkeypatch, clean_registry):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "get", boom)
    assert U.search_companies("whatever company") == ()


def test_ensure_registered_rejects_symbols_that_are_not_nse_bse(clean_registry):
    assert U.ensure_registered("AAPL") is None
    assert U.ensure_registered("TCS.NS").name == "Tata Consultancy Services"   # already known


# --- found on the deployed app -----------------------------------------------------------------
def test_missing_yahoo_info_is_filled_from_the_statements(monkeypatch, tmp_path):
    """From a shared cloud host Yahoo answers `info` with an almost empty record: the ratios the
    briefing needs must then be computed from the annual statements and the last close."""
    from data.connectors import live

    monkeypatch.setattr(live, "CACHE_DIR", str(tmp_path))
    stmt = {"currency": "INR", "years": [{"end": "2026-03-31", "revenue": 1000.0, "net_income": 200.0,
                                          "equity": 800.0, "total_debt": 80.0, "eps": 10.0}]}
    monkeypatch.setattr(live, "fetch_statements", lambda t: stmt)
    monkeypatch.setattr(live, "fetch_prices", lambda t, period="1y": {"close": [150.0]})

    class FakeTicker:
        def __init__(self, t):
            self.info = {"trailingPegRatio": None}
            self.fast_info = {"market_cap": 5.0e12}

    import sys
    import types
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=FakeTicker))
    f = live.fetch_fundamentals("X.NS")
    assert round(f["roe"], 1) == 25.0 and round(f["profit_margin"], 1) == 20.0
    assert round(f["debt_to_equity"], 1) == 10.0 and round(f["pe"], 1) == 15.0
    assert f["market_cap"] == 5.0e12 and "pe" in f["computed_fields"]
