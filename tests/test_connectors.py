"""Connectors and LLM backends, exercised with mocks: no network, no model servers."""

import datetime as dt
import json
import time

import pytest

from data.connectors import gnews, live
from llm import LLM, LLMUnavailable, OllamaLLM, get_llm


class Entry(dict):
    """feedparser entries are dicts with attribute access to `title`."""
    @property
    def title(self):
        return self["title"]


class Feed(dict):
    @property
    def entries(self):
        return self["entries"]


def _entry(title, source="Moneycontrol", when="Mon, 07 Sep 2026 07:00:00 GMT"):
    return Entry(title=f"{title} - {source}", source={"title": source}, published=when,
                 link="https://example.test/a")


@pytest.fixture
def fake_feed(monkeypatch, tmp_path):
    monkeypatch.setattr(gnews, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(gnews, "MIN_GAP_S", 0)
    calls = []

    def install(entries_for):
        import feedparser

        def parse(url, agent=None):
            calls.append(url)
            return Feed(status=200, entries=entries_for(url))
        monkeypatch.setattr(feedparser, "parse", parse)
    install.calls = calls
    return install


# ------------------------------------------------------------------- gnews
def test_gnews_parses_headlines_and_strips_the_publisher_suffix(fake_feed):
    fake_feed(lambda url: [_entry("Infosys wins a deal")])
    items = gnews.fetch_window('"Infosys"', dt.date(2026, 9, 1), dt.date(2026, 9, 7))
    assert items == [{"title": "Infosys wins a deal", "source": "Moneycontrol",
                      "url": "https://example.test/a", "seendate": "20260907T070000Z"}]


def test_gnews_results_are_cached_on_disk(fake_feed):
    fake_feed(lambda url: [_entry("Headline one")])
    a, b = dt.date(2026, 9, 1), dt.date(2026, 9, 7)
    gnews.fetch_window("q", a, b)
    gnews.fetch_window("q", a, b)
    assert len(fake_feed.calls) == 1                      # second call served from cache


def test_gnews_splits_a_full_window_so_busy_stocks_are_not_truncated(fake_feed):
    full = [_entry(f"Story {i}") for i in range(gnews.FULL)]
    fake_feed(lambda url: full if "after:2026-09-01+before:2026-09-09" in url.replace("%20", "+")
              or len(fake_feed.calls) == 1 else [_entry("Half story")])
    items = gnews.fetch_window("q", dt.date(2026, 9, 1), dt.date(2026, 9, 8))
    assert len(fake_feed.calls) == 3                      # the full window, then two halves
    assert {i["title"] for i in items} == {"Half story"}


def test_gnews_range_deduplicates_syndicated_headlines(fake_feed):
    fake_feed(lambda url: [_entry("Same story", "A"), _entry("Same story", "B")])
    out = gnews.fetch_range("q", dt.date(2026, 9, 1), dt.date(2026, 9, 14), window_days=7)
    assert [i["title"] for i in out] == ["Same story"]


def test_gnews_gives_up_after_repeated_failures(monkeypatch, tmp_path):
    import feedparser
    monkeypatch.setattr(gnews, "CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(gnews, "MIN_GAP_S", 0)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(feedparser, "parse", lambda url, agent=None: Feed(status=429, entries=[]))
    with pytest.raises(RuntimeError):
        gnews._request("q", dt.date(2026, 9, 1), dt.date(2026, 9, 2), attempts=2)


# -------------------------------------------------------------------- live
def test_live_prices_use_a_fresh_cache_without_touching_the_network(monkeypatch, tmp_path):
    monkeypatch.setattr(live, "CACHE_DIR", str(tmp_path))
    payload = {"date": ["2026-01-01"], "close": [1.0], "provenance": "yfinance"}
    live._write_cache("px_TEST_1y", payload)
    import sys
    monkeypatch.setitem(sys.modules, "yfinance", None)    # any import would now fail
    assert live.fetch_prices("TEST") == payload


def test_live_prices_return_none_when_the_source_fails(monkeypatch, tmp_path):
    import sys
    import types
    monkeypatch.setattr(live, "CACHE_DIR", str(tmp_path))
    boom = types.SimpleNamespace(Ticker=lambda t: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setitem(sys.modules, "yfinance", boom)
    assert live.fetch_prices("NOPE") is None


def test_stale_cache_is_ignored(monkeypatch, tmp_path):
    import os
    monkeypatch.setattr(live, "CACHE_DIR", str(tmp_path))
    live._write_cache("k", {"x": 1})
    old = time.time() - 13 * 3600
    os.utime(live._cache_path("k"), (old, old))
    assert live._read_cache("k", max_age_hours=12) is None


# --------------------------------------------------------------------- llm
def test_no_llm_configured_is_reported_honestly(monkeypatch):
    monkeypatch.setenv("FINRESEARCH_LLM", "none")
    llm = get_llm()
    assert not llm.available
    with pytest.raises(LLMUnavailable):
        llm.generate("s", "u")


def test_unreachable_backends_fall_back_to_none(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("FINRESEARCH_LLM", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")     # nothing listens here
    assert type(get_llm()) is LLM


def test_ollama_host_is_read_from_the_environment(monkeypatch):
    import requests

    class R:
        def raise_for_status(self): pass
        def json(self): return {"models": [{"name": "mistral:latest"}]}

    seen = {}
    monkeypatch.setattr(requests, "get", lambda url, timeout: seen.setdefault("url", url) and R())
    monkeypatch.setenv("OLLAMA_HOST", "ollama:11434")           # no scheme, as Docker users write it
    llm = OllamaLLM()
    assert seen["url"] == "http://ollama:11434/api/tags" and llm.name == "ollama/mistral:latest"


def test_ollama_generate_requests_json_mode(monkeypatch):
    import requests

    class Tags:
        def raise_for_status(self): pass
        def json(self): return {"models": [{"name": "m"}]}

    class Chat:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": ' {"ok": true} '}}

    sent = {}
    monkeypatch.setattr(requests, "get", lambda url, timeout: Tags())
    monkeypatch.setattr(requests, "post", lambda url, json, timeout: sent.update(body=json) or Chat())
    out = OllamaLLM(host="http://x").generate("sys", "usr", json_mode=True)
    assert json.loads(out) == {"ok": True} and sent["body"]["format"] == "json"


def test_groq_reasoning_models_get_room_to_think(monkeypatch):
    import sys
    import types

    captured = {}

    class Completions:
        def create(self, **kw):
            captured.update(kw)
            msg = types.SimpleNamespace(content="{}")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    class FakeGroq:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "groq", types.SimpleNamespace(Groq=FakeGroq))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    from llm import GroqLLM
    GroqLLM("openai/gpt-oss-120b").generate("s", "u", json_mode=True)
    assert captured["reasoning_effort"] == "low" and captured["max_completion_tokens"] >= 2000
    assert captured["response_format"] == {"type": "json_object"}
    captured.clear()
    GroqLLM("some/other-model").generate("s", "u")
    assert "reasoning_effort" not in captured and captured["max_tokens"] == 900
