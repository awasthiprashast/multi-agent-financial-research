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
