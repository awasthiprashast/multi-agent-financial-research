"""Unit tests.  Run: python -m pytest -q"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.orchestrator import Router, resolve_ticker           # noqa: E402
from agents.validator import VerificationAgent                    # noqa: E402
from core import Claim, extract_numbers                           # noqa: E402
from pipeline import ResearchPipeline                             # noqa: E402

PIPE = ResearchPipeline(prefer_live=False)


# ------------------------------------------------------------------ numerics
def test_number_extraction_units():
    facts = extract_numbers("Revenue was ₹240,893 cr, up 6.8%, margin -70 bps.")
    vals = {f.raw.strip() for f in facts}
    assert any("240,893" in v for v in vals)
    assert any("6.8" in v for v in vals)
    assert any("70" in v for v in vals)


def test_year_is_not_treated_as_a_quantity():
    assert extract_numbers("The financial year 2026 was strong.") == []


# -------------------------------------------------------------------- router
def test_router_selects_single_agent_for_narrow_query():
    assert Router().fit().route("what is the rsi on tcs") == ["technical"]


def test_router_expands_for_broad_query():
    picked = Router().fit().route("give me a full research report on reliance")
    assert len(picked) >= 3


def test_ticker_resolution():
    assert resolve_ticker("how is infy doing") == "INFY.NS"


# ----------------------------------------------------------------- validator
def test_genuine_claim_passes():
    rep = PIPE.run("Complete analysis of Infosys")
    assert rep.grounding_rate == 1.0


def test_altered_number_is_rejected():
    store = PIPE._prepare("TCS.NS")["store"]
    v = VerificationAgent(store)
    good = Claim.new("t", "s", "Return on equity for the year was 51.2%.")
    bad = Claim.new("t", "s", "Return on equity for the year was 74.9%.")
    good.cited_chunk_ids = [c.chunk_id for c, _ in store.query(good.text, k=2)]
    bad.cited_chunk_ids = list(good.cited_chunk_ids)
    assert v.verify(good).status == "verified"
    assert v.verify(bad).status == "flagged"


def test_fabricated_detail_is_rejected():
    store = PIPE._prepare("TCS.NS")["store"]
    c = Claim.new("t", "s", "The board approved a share buyback of ₹18,000 cr.")
    c.cited_chunk_ids = [ch.chunk_id for ch, _ in store.query(c.text, k=2)]
    assert VerificationAgent(store).verify(c).status == "flagged"


def test_entity_binding_rejects_transplanted_claim():
    store = PIPE._prepare("TCS.NS")["store"]
    c = Claim.new("t", "s",
                  "Adani Green Energy reported return on equity of 51.2%.")
    c.cited_chunk_ids = [ch.chunk_id for ch, _ in store.query(c.text, k=2)]
    assert VerificationAgent(store).verify(c).status == "flagged"


def test_revision_loop_is_bounded():
    """A claim that can never be fixed gets exactly max_retries attempts."""
    store = PIPE._prepare("TCS.NS")["store"]

    class Stubborn:
        calls = 0

        def revise(self, claim, feedback, ticker, context):
            Stubborn.calls += 1
            return Claim.new("stub", "s", "Return on equity for the year was 74.9%.",
                             claim.cited_chunk_ids)

    bad = Claim.new("stub", "s", "Return on equity for the year was 74.9%.")
    bad.cited_chunk_ids = [c.chunk_id for c, _ in store.query(bad.text, k=2)]
    v = VerificationAgent(store, max_retries=2)
    surfaced, verdicts, stats = v.verify_with_revision(
        [bad], {"stub": Stubborn()}, "TCS.NS", {})
    assert Stubborn.calls == 2
    assert surfaced == [] and stats["dropped"] == 1


def test_revision_loop_recovers_a_fixable_claim():
    store = PIPE._prepare("TCS.NS")["store"]

    class Fixer:
        def revise(self, claim, feedback, ticker, context):
            return Claim.new("fix", "s", "Return on equity for the year was 51.2%.",
                             claim.cited_chunk_ids)

    bad = Claim.new("fix", "s", "Return on equity for the year was 74.9%.")
    bad.cited_chunk_ids = [c.chunk_id for c, _ in store.query(bad.text, k=2)]
    surfaced, verdicts, stats = VerificationAgent(store).verify_with_revision(
        [bad], {"fix": Fixer()}, "TCS.NS", {})
    assert len(surfaced) == 1 and stats["revisions_successful"] == 1
    assert verdicts[surfaced[0].claim_id].status == "revised"


# ------------------------------------------------- numeric binding & direction
def _verdict(text, ticker="TCS.NS"):
    store = PIPE._prepare(ticker)["store"]
    c = Claim.new("t", "s", text)
    c.cited_chunk_ids = [ch.chunk_id for ch, _ in store.query(text, k=2)]
    return VerificationAgent(store).verify(c)


def test_direction_in_words_matches_signed_evidence():
    assert _verdict("Operating margin for the year was 24.3%, which fell 70 bps "
                    "against the prior year.").status == "verified"


def test_wrong_direction_is_rejected():
    assert _verdict("Operating margin for the year was 24.3%, which rose 70 bps "
                    "against the prior year.").status == "flagged"


def test_number_must_match_its_own_label():
    # 51.2 is the ROE; calling it the dividend yield must not be grounded.
    assert _verdict("Dividend yield was 51.2%.").status == "flagged"


def test_unicode_minus_is_parsed():
    assert extract_numbers("a movement of −70 bps")[0].value == -70


def test_entity_match_is_word_bounded():
    v = VerificationAgent(PIPE._prepare("TCS.NS")["store"])
    assert v._entities_in("In April the brilliant results arrived") == set()
    assert v._entities_in("RIL reported") == {"RELIANCE.NS"}


# ------------------------------------------------ recomputation from raw prices
def test_indicator_claim_is_checked_against_raw_prices_not_indexed_text():
    """Tamper with the indexed RSI chunk: a claim echoing the tampered value must
    still be rejected, because RSI is recomputed from the price series."""
    import copy
    ctx = PIPE._prepare("TCS.NS")
    real = ctx["indicators"]["rsi_14"]
    fake = round(real + 17.0, 1)
    store = copy.copy(ctx["store"])            # shares chunks; own recompute cache
    store._recompute_cache = None
    store.chunks = dict(ctx["store"].chunks)
    for cid, ch in list(store.chunks.items()):
        if ch.source == "derived_TCS.NS_trend":
            import dataclasses
            store.chunks[cid] = dataclasses.replace(
                ch, text=ch.text.replace(f"RSI(14) is {real}", f"RSI(14) is {fake}"))
    good = Claim.new("t", "s", f"RSI(14) is {real}, which places the stock in neutral territory.")
    bad = Claim.new("t", "s", f"RSI(14) is {fake}, which places the stock in neutral territory.")
    for c in (good, bad):
        c.cited_chunk_ids = [ch.chunk_id for ch, _ in store.query(c.text, k=2)]
    v = VerificationAgent(store)
    assert v.verify(good).status == "verified"
    assert v.verify(bad).status == "flagged"


def test_wrong_session_count_is_rejected():
    n = PIPE._prepare("TCS.NS")["indicators"]["sessions"]
    assert _verdict(f"Total price return over the {n + 7}-session window was 5.0%.").status == "flagged"


# ------------------------------------------------------ directional (no numbers)
def test_directional_claims_are_checked_against_recomputed_indicators():
    from agents.validator import _direction_conflicts
    ind = {"last_close": 110.0, "sma_20": 100.0, "sma_50": 120.0, "macd_line": 2.0,
           "macd_signal": 1.0, "macd_hist": 1.0, "rsi_14": 75.0, "period_return_pct": 8.0}
    closes = [100.0, 105.0, 110.0]
    f = lambda t: _direction_conflicts(t, ind, closes)
    assert f("The closing price is above the 20-day moving average.") == []
    assert f("The closing price is below the 20-day moving average.")
    assert f("The closing price is lower than both moving averages.")      # 110 > 100
    assert f("The MACD line is above the signal line.") == []
    assert f("The MACD line is below the signal line.")
    assert f("The histogram is negative.")
    assert f("RSI indicates overbought conditions.") == []
    assert f("RSI indicates oversold conditions.")
    assert f("The stock has a negative price return.")
    assert f("The MACD line is not below the signal line.") == []          # negation
    # a ticker's dot must not hide the claim from the rules (regression: INFY.NS)
    assert f("The RSI(14) for Infosys (INFY.NS) is below the 70 threshold, "
             "indicating neutral conditions.")                              # RSI is 75
    assert f("The RSI(14) for Infosys (INFY.NS) is above the 70 threshold, "
             "indicating overbought conditions.") == []
    ind["bb_upper"], ind["bb_lower"] = 120.0, 90.0
    assert f("The Bollinger band upper for X is higher than the current closing price.") == []
    assert f("The upper Bollinger band is lower than the closing price.")
    assert f("The last close is the lowest close in the window.")          # 110 is not min
    assert f("The closing price has decreased over the 252 trading sessions.")  # +8%
    assert f("The closing price has increased over the 252 trading sessions.") == []
    # regression: the second clause of "higher than A but lower than B" was never checked
    ind2 = dict(ind, last_close=110.0, sma_20=100.0, sma_50=90.0)          # above both
    g = lambda t: _direction_conflicts(t, ind2, closes)
    assert g("The price is higher than the 20-day moving average but lower than the "
             "50-day moving average.")                                      # 2nd clause false
    assert g("The price is higher than the 20-day moving average but higher than the "
             "50-day moving average.") == []
    # equality is checked, not waved through (found by auditing held-out LLM output)
    ind3 = dict(ind, last_close=1325.9, sma_20=1314.42, sma_50=1329.0)
    h = lambda t: _direction_conflicts(t, ind3, closes)
    assert h("The closing price is equal to the 20-day simple moving average.")        # 0.87% off
    assert h("The closing price is equal to the 50-day simple moving average.") == []  # 0.23% off
    # average as subject
    assert g("The 20-day simple moving average is lower than the current closing price.") == []
    assert g("The 20-day simple moving average is higher than the current closing price.")


def test_self_contradictory_macd_claim_is_withheld():
    ind = PIPE._prepare("TCS.NS")["indicators"]
    above = ind["macd_line"] > ind["macd_signal"]
    wrong = "below" if above else "above"
    v = _verdict(f"The MACD line is {wrong} the signal line, which is notable for TCS.NS.")
    assert v.status == "flagged" and "recomputed" in v.reason


# ------------------------------------------------------------- determinism
def test_retrieval_ties_are_broken_by_chunk_id():
    from core import Chunk
    from rag.store import EvidenceStore
    mk = lambda i: Chunk(chunk_id=f"c{i}", text="t", source="s", doc_type="d",
                         ticker="T", published="")
    cands = [(mk(3), 0.5), (mk(1), 0.5), (mk(2), 0.5), (mk(0), 0.9)]
    assert [c.chunk_id for c, _ in EvidenceStore._top_k(cands, 3)] == ["c0", "c1", "c2"]


def test_chroma_search_matches_exact_ranking():
    import numpy as np
    from rag.store import EvidenceStore
    st = PIPE._prepare("TCS.NS")["store"]
    q = "Return on equity for the year was 51.2%."
    qv = st.embedder.encode([q])[0]
    exact = EvidenceStore._top_k(
        [(st.chunks[c], float(s)) for c, s in zip(st._ids, st._mat @ qv)], 5)
    assert [c.chunk_id for c, _ in st.query(q, k=5)] == [c.chunk_id for c, _ in exact]


# ----------------------------------------------------------------- reporting
def test_grounding_rate_counts_revised_claims():
    from core import Report, Verdict
    r = Report("q", "T", {}, {
        "a": Verdict("a", 0.9, 1, 1, True, status="verified"),
        "b": Verdict("b", 0.8, 1, 1, True, status="revised"),
        "c": Verdict("c", 0.2, 0, 0, False, status="flagged"),
    }, [], {})
    assert r.surfaced_count == 2 and abs(r.grounding_rate - 2 / 3) < 1e-9


def test_price_series_is_deterministic_across_calls():
    from data.corpus import _price_series
    assert _price_series("TCS.NS")["close"] == _price_series("TCS.NS")["close"]


# ------------------------------------------------------------------ pipeline
def test_nothing_unverified_reaches_the_report():
    rep = PIPE.run("Should I look at TCS for a 1-year hold?")
    for claims in rep.sections.values():
        for c in claims:
            assert rep.verdicts[c.claim_id].status in ("verified", "revised")


def test_report_carries_disclaimer():
    rep = PIPE.run("How volatile is Reliance?")
    assert "not investment advice" in PIPE.markdown(rep).lower()


def test_every_surfaced_claim_has_a_citation():
    rep = PIPE.run("Fundamentals and news flow for TCS")
    for claims in rep.sections.values():
        for c in claims:
            assert c.cited_chunk_ids
