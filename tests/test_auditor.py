"""The text auditor: someone else's text, checked sentence by sentence."""

import pytest

from agents.auditor import (
    CONTRADICTED,
    UNCHECKABLE,
    VERIFIED,
    audit_text,
    split_sentences,
)
from data.universe import UnknownCompanyError
from pipeline import ResearchPipeline


@pytest.fixture(scope="module")
def pipe():
    return ResearchPipeline(prefer_live=False, analysts="template")


def _by_status(result):
    return {s.text: s for s in result.sentences}


def test_splitter_keeps_decimals_and_abbreviations_together():
    parts = split_sentences("Margin was 24.3% in FY25. Price is Rs. 100 today.\n- RSI is 50.\n2) Done now.")
    assert parts == ["Margin was 24.3% in FY25.", "Price is Rs. 100 today.", "RSI is 50.", "Done now."]


def test_true_number_is_verified_and_a_changed_digit_is_contradicted(pipe):
    good = "Operating margin for the year was 24.3%, a movement of -70 bps against the prior year."
    bad = good.replace("24.3%", "26.3%")
    r = audit_text(f"TCS update. {good} {bad}", pipe)
    sents = _by_status(r)
    assert sents[good].status == VERIFIED
    assert sents[bad].status == CONTRADICTED
    wrong = [n for n in sents[bad].numbers if not n.matched]
    assert [n.raw for n in wrong] == ["26.3%"]
    assert wrong[0].actual is not None            # shows what the data holds


def test_forecasts_and_prose_are_uncheckable_never_verified(pipe):
    r = audit_text("TCS had a strong year. TCS shares will reach Rs. 5,000 by FY27.", pipe)
    assert [s.status for s in r.sentences] == [UNCHECKABLE, UNCHECKABLE]
    assert r.accuracy is None and r.coverage == 0.0


def test_period_labels_are_not_treated_as_numbers(pipe):
    r = audit_text("In Q2 FY25 TCS operating margin for the year was 24.3%.", pipe)
    assert [n.raw for n in r.sentences[0].numbers] == ["24.3%"]


def test_direction_is_checked_against_the_raw_prices(pipe):
    """Exactly one of 'above' / 'below' is true; the auditor must tell them apart."""
    r = audit_text("TCS closed above its 50-day average.\nTCS closed below its 50-day average.",
                   pipe, ticker="TCS.NS")
    assert sorted(s.status for s in r.sentences) == [CONTRADICTED, VERIFIED]
    wrong = next(s for s in r.sentences if s.status == CONTRADICTED)
    assert wrong.directions_contradicted


def test_multi_company_sentences_are_not_checked(pipe):
    r = audit_text("TCS and Infosys both reported an operating margin of 24.3%.", pipe)
    assert r.sentences[0].status == UNCHECKABLE


def test_text_naming_no_company_needs_a_ticker(pipe):
    with pytest.raises(UnknownCompanyError):
        audit_text("Operating margin was 24.3% for the year.", pipe)
    r = audit_text("Operating margin for the year was 24.3%.", pipe, ticker="TCS.NS")
    assert r.sentences[0].status == VERIFIED


def test_rejects_empty_input(pipe):
    with pytest.raises(ValueError):
        audit_text("   \n ", pipe, ticker="TCS.NS")


def test_200_day_average_is_not_confused_with_the_20_day_average():
    from agents.validator import _direction_conflicts
    ind = {"last_close": 90.0, "sma_20": 80.0, "sma_50": 85.0, "sma_200": 100.0,
           "rsi_14": 55.0, "macd_line": 0, "macd_signal": 0, "macd_hist": 0,
           "bb_upper": 0, "bb_lower": 0, "period_return_pct": 0}
    confirmed: list = []
    # above the 20-day (80) but below the 200-day (100): only the 200-day statement is false
    assert _direction_conflicts("The stock closed above its 20-day average.", ind, [], confirmed) == []
    assert confirmed and "sma_20" in confirmed[0]
    wrong = _direction_conflicts("The stock closed above its 200-day average.", ind, [])
    assert wrong and "sma_200" in wrong[0] and "sma_20 " not in wrong[0]


def test_rsi_period_is_not_a_value(pipe):
    """'RSI(14)' holds the number 14 as a period; 'RSI is 14' must not be grounded by it."""
    r = audit_text("TCS RSI is 14.", pipe, ticker="TCS.NS")
    assert r.sentences[0].status == CONTRADICTED
    assert all(n.actual != "14" for n in r.sentences[0].numbers)


def test_a_50_day_average_cannot_be_grounded_by_the_20_day_value(pipe):
    ind = pipe._prepare("TCS.NS")["indicators"]
    swapped = audit_text(f"TCS 50-day simple moving average is ₹{ind['sma_20']}.", pipe, ticker="TCS.NS")
    right = audit_text(f"TCS 50-day simple moving average is ₹{ind['sma_50']}.", pipe, ticker="TCS.NS")
    assert swapped.sentences[0].status == CONTRADICTED
    assert right.sentences[0].status == VERIFIED


def test_a_partly_checkable_sentence_is_never_fully_correct(pipe):
    ind = pipe._prepare("TCS.NS")["indicators"]
    word = "above" if ind["macd_line"] > ind["macd_signal"] else "below"
    r = audit_text(f"TCS MACD line is {word} its signal line, after crossing it on October 15.",
                   pipe, ticker="TCS.NS")
    assert r.sentences[0].status == "unsupported"
    assert "only part" in r.sentences[0].reason
    plain = audit_text(f"TCS MACD line is {word} its signal line.", pipe, ticker="TCS.NS")
    assert plain.sentences[0].status == VERIFIED


def test_us_is_not_a_sentence_end():
    parts = split_sentences("Risks include a slowdown in U.S. IT spending. Margins are 24.3%.")
    assert parts == ["Risks include a slowdown in U.S. IT spending.", "Margins are 24.3%."]


# --- found by the real-text benchmark (development companies only) ---------------------------
@pytest.fixture(scope="module")
def fpipe():
    """Real TCS data frozen on 2026-10-27 (ROE 45.6 %, profit margin 18.2 %, P/E 17.8,
    market cap 917,185 cr, volatility 30.1 %)."""
    from benchmark.frozen import FrozenPipeline
    return FrozenPipeline()


def _status(pipe, text):
    return audit_text(text, pipe, ticker="TCS.NS").sentences[0]


def test_a_rounded_or_approximate_figure_is_not_a_wrong_one(fpipe):
    # TCS: ROE 45.6 %, profit margin 18.2 %, trailing P/E 17.8
    assert _status(fpipe, "ROE sits around 45% and net margins are 18%.").status == VERIFIED
    assert _status(fpipe, "ROE sits around 40% and net margins are 18%.").status == CONTRADICTED


def test_lakh_crore_and_trillion_are_real_units(fpipe):
    from core import extract_numbers
    (n,) = extract_numbers("a market cap of roughly ₹9.2 lakh crore")
    assert n.unit == "lakh crore" and n.normalised == 9.2e12
    assert _status(fpipe, "The market cap is roughly ₹9.2 lakh crore.").status == VERIFIED
    assert _status(fpipe, "The market cap is roughly ₹92 lakh crore.").status == CONTRADICTED


def test_a_hyphen_between_digits_is_a_range_not_a_minus():
    from core import extract_numbers
    assert [n.value for n in extract_numbers("ROE of 14-15% in FY-2027")] == [14.0, 15.0]
    assert [n.value for n in extract_numbers("a movement of -70 bps")] == [-70.0]


def test_the_letters_rsi_inside_another_word_are_not_the_rsi(pipe):
    s = _status(pipe, "The stock is in a persistent downtrend over the last 246 sessions.")
    assert not s.directions_contradicted


def test_nearing_overbought_is_not_a_claim_that_the_rsi_is_overbought(pipe):
    ind = pipe._prepare("TCS.NS")["indicators"]
    s = _status(pipe, "The RSI is nearing overbought territory.")
    assert not s.directions_contradicted              # whatever the RSI is, nothing was asserted
    assert ind["rsi_14"] > 0


def test_a_hedge_does_not_hide_the_facts_beside_it(fpipe):
    s = _status(fpipe, "Annualised volatility of 30.1% could rise further if markets sell off.")
    assert s.status == VERIFIED and "30.1%" in s.explanation


def test_forward_quarterly_and_peer_figures_are_not_compared_with_annual_ones(pipe):
    s = _status(pipe, "TCS trades at a forward P/E of roughly 28x against a sector average of 24x.")
    assert s.status != CONTRADICTED and s.status != VERIFIED
    q = _status(pipe, "Net profit for the latest quarter was ₹18,500 crore.")
    assert q.status != CONTRADICTED


def test_a_stated_price_level_is_checked_against_the_last_close(pipe):
    ind = pipe._prepare("TCS.NS")["indicators"]
    assert _status(pipe, "TCS has recently consolidated near ₹3,800.").status == CONTRADICTED
    near = round(ind["last_close"])
    assert _status(pipe, f"TCS has recently consolidated near ₹{near:,}.").status == VERIFIED


def test_a_condition_is_not_a_claim_about_today(pipe):
    s = _status(pipe, "TCS needs caution until the price stabilises above its 50-day average.")
    assert not s.directions_contradicted


def test_roe_is_not_matched_to_volatility_or_debt_to_equity(fpipe):
    # frozen TCS: ROE 45.6 %, annualised volatility 30.1 %, debt-to-equity 9.8 %
    s = _status(fpipe, "ROE sits around 30%.")
    assert s.status == CONTRADICTED and all(n.actual != "30.1%" for n in s.numbers)
    assert _status(fpipe, "Return on equity, computed as net income over year-end equity, was 45.9%.").status == VERIFIED


def test_a_wrong_price_level_is_explained_against_the_last_close_and_corrected(fpipe):
    r = audit_text("TCS has consolidated near ₹3,800.", fpipe, ticker="TCS.NS").sentences[0]
    assert r.status == CONTRADICTED
    assert "last close is ₹2,530" in r.explanation and "stated 3,800" not in r.explanation
    assert r.correction and "₹2,530.00" in r.correction


def test_a_big_market_cap_is_shown_in_lakh_crore(fpipe):
    r = audit_text("The market cap is roughly ₹1.06 lakh crore.", fpipe, ticker="TCS.NS").sentences[0]
    assert r.status == CONTRADICTED and "lakh crore" in r.explanation


def test_a_fiscal_year_label_is_not_two_numbers():
    from core import extract_numbers
    assert [n.raw for n in extract_numbers("FY 2025-26 revenue grew 12% YoY")] == ["12%"]
    assert [n.raw for n in extract_numbers("margin was 24.3% in 2025/2026")] == ["24.3%"]
