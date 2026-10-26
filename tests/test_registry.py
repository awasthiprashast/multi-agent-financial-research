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
