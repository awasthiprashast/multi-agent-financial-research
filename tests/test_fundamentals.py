"""Real-fundamentals path: statements -> evidence chunk -> claims -> verifier (no network)."""

from agents.analysts import FundamentalAgent
from agents.validator import VerificationAgent
from core import Claim
from data.corpus import statements_chunk
from pipeline import ResearchPipeline
from rag.store import EvidenceStore

CR = 1e7
STMT = {
    "currency": "INR",
    "years": [
        {"end": "2026-03-31", "revenue": 267021 * CR, "operating_income": 67022 * CR,
         "net_income": 49210 * CR, "eps": 136.01, "total_debt": 11283 * CR,
         "equity": 107240 * CR, "total_assets": 182372 * CR},
        {"end": "2025-03-31", "revenue": 255324 * CR, "operating_income": 62293 * CR,
         "net_income": 48553 * CR, "eps": 134.19, "total_debt": 9392 * CR,
         "equity": 94756 * CR, "total_assets": 159629 * CR},
    ],
}


def _store():
    chunk = statements_chunk("TCS.NS", STMT)
    return EvidenceStore(use_chroma=False).index([chunk]), chunk


def test_ratios_are_computed_in_python_not_copied():
    _, chunk = _store()
    t = chunk.text
    assert "Total revenue is ₹267,021 cr." in t
    assert "Revenue growth over the prior fiscal year is 4.6%." in t      # 267021/255324-1
    assert "Operating margin is 25.1%." in t                               # 67022/267021
    assert "Net margin is 18.4%." in t
    assert "Debt-to-equity ratio is 0.11." in t                            # 11283/107240
    assert "Return on equity is 45.9%." in t                               # 49210/107240
    assert chunk.meta["provenance"] == "yfinance-statements"


def test_a_bank_without_operating_income_still_yields_a_chunk():
    bank = {"currency": "INR", "years": [
        {"end": "2026-03-31", "revenue": 192566 * CR, "operating_income": None,
         "net_income": 70479 * CR, "eps": 45.89, "total_debt": 664364 * CR,
         "equity": 816739 * CR, "total_assets": None}]}
    t = statements_chunk("HDFCBANK.NS", bank).text
    assert "Operating margin" not in t and "Net margin is 36.6%." in t
    assert "Debt-to-equity" not in t           # meaningless for a lender: deposits count as debt


def test_non_inr_or_empty_statements_are_rejected():
    assert statements_chunk("X.NS", {"currency": "USD", "years": [STMT["years"][0]]}) is None
    assert statements_chunk("X.NS", {"currency": "INR", "years": []}) is None
    assert statements_chunk("X.NS", None) is None


def test_fundamental_agent_claims_from_statements_all_verify():
    store, chunk = _store()
    claims = FundamentalAgent(store)._live_claims("TCS.NS")
    texts = " ".join(c.text for c in claims)
    assert "₹267,021 cr" in texts and "4.6%" in texts and "25.1%" in texts and "0.11" in texts
    v = VerificationAgent(store)
    for c in claims:
        assert c.cited_chunk_ids == [chunk.chunk_id]
        assert v.verify(c).status == "verified", c.text


def test_a_wrong_financial_number_is_rejected():
    store, chunk = _store()
    v = VerificationAgent(store)
    for text in ("Total revenue for the fiscal year ended 2026-03-31 was ₹290,000 cr.",
                 "The operating margin for the fiscal year ended 2026-03-31 was 31.5%.",
                 "Return on equity was 61.0%."):
        c = Claim.new("t", "s", text, [chunk.chunk_id])
        assert v.verify(c).status == "flagged", text


def test_net_margin_cannot_be_passed_off_as_operating_margin():
    store, chunk = _store()
    c = Claim.new("t", "s", "The operating margin was 18.4%.", [chunk.chunk_id])    # that is net
    assert VerificationAgent(store).verify(c).status == "flagged"


def test_offline_mode_rejects_a_company_without_a_snapshot():
    import pytest

    from data.universe import UnknownCompanyError
    pipe = ResearchPipeline(prefer_live=False)
    with pytest.raises(UnknownCompanyError):
        pipe.run("full report on Wipro")
    with pytest.raises(UnknownCompanyError):
        pipe.run("what is the weather")


def test_company_resolution_uses_the_first_company_mentioned():
    from data.universe import find_tickers, resolve_ticker
    assert resolve_ticker("Compare Reliance and TCS") == "RELIANCE.NS"
    assert find_tickers("Compare Reliance and TCS") == ["RELIANCE.NS", "TCS.NS"]
    assert resolve_ticker("how is HDFC Bank doing") == "HDFCBANK.NS"
    assert resolve_ticker("what happened in April") is None            # 'ril' inside 'April'
    assert resolve_ticker("nothing here") is None
