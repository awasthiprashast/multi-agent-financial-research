"""Live news wiring (Google News mocked): only headlines naming the company reach the store."""

from data import corpus
from data.connectors import gnews
from data.universe import mentions


def _raw(title, day="20260915T070000Z", source="Moneycontrol"):
    return {"title": title, "source": source, "url": "https://x.test/" + title[:6], "seendate": day}


def test_only_headlines_naming_the_company_are_kept(monkeypatch):
    monkeypatch.setattr(gnews, "fetch_window", lambda *a, **k: [
        _raw("Infosys wins a large deal in Europe"),
        _raw("Nifty IT slips as tech stocks fall", "20260916T070000Z"),       # no company
        _raw("TCS and Infosys lead the rally", "20260917T070000Z"),
    ])
    chunks = corpus._live_news_chunks("INFY.NS", rss_fallback=lambda q: [])
    assert [c.text for c in chunks] == ["TCS and Infosys lead the rally",
                                        "Infosys wins a large deal in Europe"]   # newest first
    assert all(c.doc_type == "news" and c.ticker == "INFY.NS" for c in chunks)
    assert chunks[0].published == "2026-09-17" and chunks[0].meta["provenance"] == "google-news-rss"


def test_news_is_capped(monkeypatch):
    many = [_raw(f"Infosys story number {i}", f"202609{(i % 28) + 1:02d}T070000Z") for i in range(120)]
    monkeypatch.setattr(gnews, "fetch_window", lambda *a, **k: many)
    assert len(corpus._live_news_chunks("INFY.NS", lambda q: [])) == corpus.LIVE_NEWS_MAX


def test_falls_back_to_plain_rss_when_google_news_fails(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("blocked")
    monkeypatch.setattr(gnews, "fetch_window", boom)
    rss = lambda q: [{"title": "Wipro results", "summary": "Profit rose.", "link": "https://r.test/1",
                      "published": "2026-09-01"}]
    chunks = corpus._live_news_chunks("WIPRO.NS", rss)
    assert len(chunks) == 1 and chunks[0].meta["provenance"] == "rss-live"


def test_no_news_at_all_yields_no_chunks_not_an_error(monkeypatch):
    monkeypatch.setattr(gnews, "fetch_window", lambda *a, **k: [])
    assert corpus._live_news_chunks("WIPRO.NS", lambda q: None) == []


def test_company_mention_is_whole_word():
    assert mentions("ITC shares rise on cigarette volume", "ITC.NS")
    assert not mentions("Switch to digital kitchens accelerates", "ITC.NS")
    assert mentions("State Bank of India raises lending rates", "SBIN.NS")
    assert not mentions("Infosys wins deal", "TCS.NS")
    assert not mentions("anything", "UNKNOWN.NS")
