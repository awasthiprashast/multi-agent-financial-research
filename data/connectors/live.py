"""
data/connectors/live.py — real, free data sources.

These are the production connectors. They require outbound network access and
are the path used on a normal laptop / on the deployed instance. When any of
them is unavailable, `data.corpus.load_corpus()` transparently falls back to the
frozen offline snapshot so the system — and the evaluation harness — still runs.

All sources are free tier or public:
  prices        yfinance (Yahoo), nsepython for NSE-specific series
  fundamentals  Screener.in (polite, cached), SEC EDGAR for US issuers
  news          RSS: Moneycontrol, Economic Times, LiveMint
  macro         RBI data releases / FRED
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from typing import Dict, List, Optional

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/business.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.livemint.com/rss/markets",
]

USER_AGENT = "VIT-BCSE497J-ProjectI research assistant (academic use)"


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.json")


def _read_cache(key: str, max_age_hours: float = 12.0) -> Optional[dict]:
    p = _cache_path(key)
    if not os.path.exists(p):
        return None
    if (time.time() - os.path.getmtime(p)) > max_age_hours * 3600:
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_cache(key: str, payload: dict) -> None:
    with open(_cache_path(key), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


# ---------------------------------------------------------------- prices
def fetch_prices(ticker: str, period: str = "1y") -> Optional[Dict[str, list]]:
    """Daily OHLCV via yfinance. Returns None if the network is unavailable."""
    cached = _read_cache(f"px_{ticker}_{period}")
    if cached:
        return cached
    try:
        import yfinance as yf

        hist = yf.Ticker(ticker).history(period=period)
        if hist is None or len(hist) == 0:
            return None
        payload = {
            "date": [d.strftime("%Y-%m-%d") for d in hist.index],
            "open": [float(x) for x in hist["Open"]],
            "high": [float(x) for x in hist["High"]],
            "low": [float(x) for x in hist["Low"]],
            "close": [float(x) for x in hist["Close"]],
            "volume": [float(x) for x in hist["Volume"]],
            "provenance": "yfinance",
            "retrieved": _dt.date.today().isoformat(),
        }
        _write_cache(f"px_{ticker}_{period}", payload)
        return payload
    except Exception:
        return None


# ----------------------------------------------------------- fundamentals
def fetch_fundamentals(ticker: str) -> Optional[Dict[str, float]]:
    """Headline fundamentals via yfinance `info` (free, no key)."""
    cached = _read_cache(f"fund_{ticker}")
    if cached:
        return cached
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        payload = {
            "pe": info.get("trailingPE"),
            "roe": (info.get("returnOnEquity") or 0) * 100 or None,
            "debt_to_equity": info.get("debtToEquity"),
            "profit_margin": (info.get("profitMargins") or 0) * 100 or None,
            "revenue": info.get("totalRevenue"),
            "market_cap": info.get("marketCap"),
            "provenance": "yfinance.info",
            "retrieved": _dt.date.today().isoformat(),
        }
        _write_cache(f"fund_{ticker}", payload)
        return payload
    except Exception:
        return None


# ------------------------------------------------------------------ news
def fetch_news(query: str, limit: int = 25) -> Optional[List[Dict[str, str]]]:
    """Headlines from public RSS feeds, filtered on the query term."""
    cached = _read_cache(f"news_{query}")
    if cached:
        return cached["items"]
    try:
        import feedparser

        items: List[Dict[str, str]] = []
        for url in RSS_FEEDS:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                blob = f"{entry.get('title', '')} {entry.get('summary', '')}"
                if query.lower().split(".")[0] in blob.lower():
                    items.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", "")[:600],
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "source": url,
                    })
                if len(items) >= limit:
                    break
        if not items:
            return None
        _write_cache(f"news_{query}", {"items": items})
        return items
    except Exception:
        return None


# ----------------------------------------------------------------- EDGAR
def fetch_edgar_facts(cik: str) -> Optional[dict]:
    """SEC XBRL company facts — free, structured, official (US issuers)."""
    cached = _read_cache(f"edgar_{cik}")
    if cached:
        return cached
    try:
        import requests

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik.zfill(10)}.json"
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        payload = r.json()
        _write_cache(f"edgar_{cik}", payload)
        return payload
    except Exception:
        return None


def network_available() -> bool:
    try:
        import requests

        requests.get("https://query1.finance.yahoo.com", timeout=5)
        return True
    except Exception:
        return False
