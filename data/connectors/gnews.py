"""
data/connectors/gnews.py — dated news headlines from Google News RSS search.

Plain RSS feeds expose only the latest few items. Google News RSS accepts a search
query with `after:` / `before:` operators, so a year of headlines can be walked in
date windows. Each feed returns at most ~100 items; a window that comes back full
is split in half and re-queried so busy stocks are not silently truncated.

What it returns: the headline, the publisher and the publication *date* (Google
normalises the time of day, so only the date is reliable). Headlines — not article
bodies — are what the sentiment study scores. Requests are spaced and cached.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import time
import urllib.parse as up
from email.utils import parsedate_to_datetime
from typing import Dict, List

log = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "gnews")
USER_AGENT = "finresearch-assistant academic research (BCSE497J)"
MIN_GAP_S = 1.5
FULL = 95                      # a feed this long was probably truncated at ~100
_last_call = 0.0


def _cache_file(query: str, start: dt.date, end: dt.date) -> str:
    key = hashlib.sha1(f"{query}|{start}|{end}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{key}.json")


def _request(query: str, start: dt.date, end: dt.date, attempts: int = 5) -> List[Dict]:
    """One feed for [start, end] inclusive. `before:` is exclusive, so add a day."""
    global _last_call
    import feedparser

    q = f"{query} after:{start.isoformat()} before:{(end + dt.timedelta(days=1)).isoformat()}"
    url = ("https://news.google.com/rss/search?q=" + up.quote(q)
           + "&hl=en-IN&gl=IN&ceid=IN:en")
    for attempt in range(attempts):
        wait = MIN_GAP_S - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            feed = feedparser.parse(url, agent=USER_AGENT)
        except Exception as exc:
            log.warning("gnews request failed (%s)", exc)
            time.sleep(5 * (attempt + 1))
            continue
        status = feed.get("status", 0)
        if status == 200:
            out = []
            for e in feed.entries:
                src = (e.get("source") or {}).get("title", "")
                title = e.get("title", "").strip()
                if src and title.endswith(" - " + src):
                    title = title[: -len(src) - 3].strip()
                try:
                    when = parsedate_to_datetime(e.get("published", ""))
                    seen = when.strftime("%Y%m%dT%H%M%SZ")
                except Exception:
                    continue
                if title:
                    out.append({"title": title, "source": src, "url": e.get("link", ""),
                                "seendate": seen})
            return out
        time.sleep(10 * (attempt + 1))        # 429 / 503: back off
    raise RuntimeError(f"gnews: giving up on {q}")


def fetch_window(query: str, start: dt.date, end: dt.date) -> List[Dict]:
    path = _cache_file(query, start, end)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    items = _request(query, start, end)
    if len(items) >= FULL and end > start:
        mid = start + (end - start) // 2
        items = fetch_window(query, start, mid) + fetch_window(query, mid + dt.timedelta(days=1), end)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(items, fh)
    return items


def fetch_range(query: str, start: dt.date, end: dt.date, window_days: int = 7,
                progress=None) -> List[Dict]:
    seen, out = set(), []
    cur = start
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=window_days - 1), end)
        for item in fetch_window(query, cur, nxt):
            key = item["title"].lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        if progress:
            progress(cur, len(out))
        cur = nxt + dt.timedelta(days=1)
    return out
