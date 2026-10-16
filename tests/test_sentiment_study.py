"""Sentiment-study plumbing: session mapping, relevance filter, headline parsing."""

from eval.llm_eval import _bad_numbers
from eval.sentiment_study import UNIVERSE, _corr, _relevant, _trading_day

DAYS = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18", "2026-09-21"]


def test_midday_headline_maps_to_same_session():
    assert _trading_day("20260915T070000Z", DAYS) == "2026-09-15"      # 12:30 IST


def test_after_close_headline_maps_to_next_session():
    assert _trading_day("20260915T120000Z", DAYS) == "2026-09-16"      # 17:30 IST


def test_weekend_headline_maps_to_next_trading_day():
    assert _trading_day("20260919T070000Z", DAYS) == "2026-09-21"      # Saturday


def test_headline_after_last_session_is_dropped():
    assert _trading_day("20260921T120000Z", DAYS) is None


def test_company_must_be_named_in_the_headline():
    aliases = UNIVERSE["INFY.NS"][1]
    assert _relevant("Infosys wins a large European deal", aliases)
    assert not _relevant("Nifty IT index slips as tech stocks fall", aliases)


def test_alias_match_is_whole_word():
    aliases = UNIVERSE["ITC.NS"][1]
    assert _relevant("ITC shares rise on cigarette volume growth", aliases)
    assert not _relevant("Switch to digital kitchens accelerates", aliases)   # 'itc' in 'switch'


def test_correlation_of_too_few_points_is_not_reported():
    assert _corr([0.1, 0.2], [0.3, 0.1])["pearson"] is None


def test_oracle_ignores_dates_and_units_but_catches_a_wrong_number():
    truth = [24.3, -70.0, 240893.0]
    assert _bad_numbers("Margin 24.3%, a movement of -70 bps on 2026-04-18.", truth) == []
    assert _bad_numbers("Revenue was ₹240,893 cr.", truth) == []
    assert _bad_numbers("Margin was 31.7%.", truth) == ["31.7%"]
