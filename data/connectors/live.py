"""
data/connectors/live.py — real, free data sources.

Live connectors. They need outbound network access. When a source is
unavailable the failure is logged and None is returned, so the caller can fall
back to the frozen offline snapshot.

All sources are free tier or public:
  prices        yfinance (Yahoo)
  fundamentals  yfinance annual statements + headline ratios
  news          generic RSS (Moneycontrol, Economic Times, LiveMint); the main
                headline path is data/connectors/gnews.py (Google News)
  EDGAR         SEC XBRL company facts (US issuers; not used for NSE names)
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import time
from typing import Dict, List, Optional

log = logging.getLogger("live")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

RSS_FEEDS = [
    "https://www.moneycontrol.com/rss/business.xml",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.livemint.com/rss/markets",
]

USER_AGENT = "finresearch-assistant research prototype"


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
def _drop_bad_rows(payload: dict) -> dict:
    """Yahoo sometimes returns today's unfinished bar with a NaN (or zero) price. One such
    row turns every indicator into NaN, so rows with a missing OHLC value are removed."""
    keys = [k for k in ("open", "high", "low", "close") if k in payload]
    keep = [i for i in range(len(payload["close"]))
            if all(payload[k][i] == payload[k][i] and payload[k][i] > 0 for k in keys)]
    if len(keep) != len(payload["close"]):
        for k in ("date", "open", "high", "low", "close", "volume"):
            if k in payload:
                payload[k] = [payload[k][i] for i in keep]
    return payload


def fetch_prices(ticker: str, period: str = "1y") -> Optional[Dict[str, list]]:
    """Daily OHLCV via yfinance. Returns None if the network is unavailable."""
    cached = _read_cache(f"px_{ticker}_{period}")
    if cached:
        return _drop_bad_rows(cached)
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
        payload = _drop_bad_rows(payload)
        if not payload["close"]:
            return None
        _write_cache(f"px_{ticker}_{period}", payload)
        return payload
    except Exception as exc:
        log.warning("fetch_prices failed: %s: %s", type(exc).__name__, exc)
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
    except Exception as exc:
        log.warning("fetch_fundamentals failed: %s: %s", type(exc).__name__, exc)
        return None


_STATEMENT_ROWS = {
    "income": {"revenue": ["Total Revenue", "Operating Revenue"],
               "operating_income": ["Operating Income"],
               "net_income": ["Net Income", "Net Income Common Stockholders"],
               "eps": ["Basic EPS", "Diluted EPS"]},
    "balance": {"total_debt": ["Total Debt"],
                "equity": ["Stockholders Equity", "Common Stock Equity"],
                "total_assets": ["Total Assets"]},
}


def _row(frame, names, col):
    for n in names:
        if n in frame.index:
            v = frame.at[n, col]
            if v == v and v is not None:        # NaN check
                return float(v)
    return None


def fetch_statements(ticker: str) -> Optional[Dict]:
    """Annual income statement and balance sheet (latest two fiscal years) via yfinance.

    Returns {"currency", "years": [{"end": "2026-03-31", revenue, operating_income,
    net_income, eps, total_debt, equity, total_assets}, ...]} newest first, with None
    for rows a company does not report (banks have no operating income). Real data from
    Yahoo's statements, not parsed from the companies' own annual-report PDFs."""
    cached = _read_cache(f"stmt_{ticker}", max_age_hours=24.0)
    if cached:
        return cached
    try:
        import yfinance as yf

        tk = yf.Ticker(ticker)
        inc, bal = tk.income_stmt, tk.balance_sheet
        if inc is None or len(inc.columns) == 0:
            return None
        years = []
        for col in list(inc.columns)[:2]:
            end = col.strftime("%Y-%m-%d")
            row = {"end": end}
            for key, names in _STATEMENT_ROWS["income"].items():
                row[key] = _row(inc, names, col)
            bal_col = next((c for c in bal.columns if c.strftime("%Y-%m-%d") == end), None) \
                if bal is not None and len(bal.columns) else None
            for key, names in _STATEMENT_ROWS["balance"].items():
                row[key] = _row(bal, names, bal_col) if bal_col is not None else None
            years.append(row)
        if years[0].get("revenue") is None and years[0].get("net_income") is None:
            return None
        info = tk.info or {}
        payload = {"currency": info.get("financialCurrency") or info.get("currency"),
                   "years": years, "provenance": "yfinance.income_stmt+balance_sheet",
                   "retrieved": _dt.date.today().isoformat()}
        _write_cache(f"stmt_{ticker}", payload)
        return payload
    except Exception as exc:
        log.warning("fetch_statements failed: %s: %s", type(exc).__name__, exc)
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
    except Exception as exc:
        log.warning("fetch_news failed: %s: %s", type(exc).__name__, exc)
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
    except Exception as exc:
        log.warning("fetch_edgar_facts failed: %s: %s", type(exc).__name__, exc)
        return None


def network_available() -> bool:
    try:
        import requests

        requests.get("https://query1.finance.yahoo.com", timeout=5)
        return True
    except Exception:
        return False
