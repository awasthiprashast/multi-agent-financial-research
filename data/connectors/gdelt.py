"""
data/connectors/gdelt.py — historical news headlines from the GDELT DOC 2.0 API.

RSS feeds only expose the latest couple of dozen items, which is useless for a
study that needs months of headlines. GDELT is free, needs no key and supports
date-windowed search. It returns headline, URL, domain and first-seen time (not
article bodies), so what we score is the headline — the standard unit in
financial-news sentiment work.

GDELT asks for at most one request every five seconds; this module spaces its
calls, backs off on HTTP 429, and caches every window on disk so a re-run is free.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import os
import time
from typing import Dict, Iterator, List, Tuple

log = logging.getLogger(__name__)

API = "https://api.gdeltproject.org/api/v2/doc/doc"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "cache", "gdelt")
USER_AGENT = "finresearch-assistant academic research (BCSE497J)"
MIN_GAP_S = 6.0
_last_call = 0.0


def _cache_file(query: str, start: str, end: str) -> str:
    key = hashlib.sha1(f"{query}|{start}|{end}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{key}.json")


def windows(start: dt.date, end: dt.date, days: int) -> Iterator[Tuple[dt.date, dt.date]]:
    cur = start
    while cur <= end:
        nxt = min(cur + dt.timedelta(days=days - 1), end)
        yield cur, nxt
        cur = nxt + dt.timedelta(days=1)


def fetch_window(query: str, start: dt.date, end: dt.date,
                 max_records: int = 250, attempts: int = 6) -> List[Dict[str, str]]:
    """Headlines matching `query` first seen between `start` and `end` (inclusive)."""
    global _last_call
    import requests

    s, e = start.strftime("%Y%m%d000000"), end.strftime("%Y%m%d235959")
    path = _cache_file(query, s, e)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    params = {"query": f"{query} sourcelang:eng", "mode": "artlist", "format": "json",
              "maxrecords": max_records, "startdatetime": s, "enddatetime": e,
              "sort": "datedesc"}
    for attempt in range(attempts):
        wait = MIN_GAP_S - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
        try:
            r = requests.get(API, params=params, timeout=45,
                             headers={"User-Agent": USER_AGENT})
        except Exception as exc:
            log.warning("GDELT request failed (%s); retrying", exc)
            time.sleep(8 * (attempt + 1))
            continue
        if r.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        try:
            articles = r.json().get("articles", []) or []
        except Exception:
            # GDELT answers an empty window with an empty body or plain text
            if r.status_code == 200 and len(r.text.strip()) < 5:
                articles = []
            else:
                time.sleep(8 * (attempt + 1))
                continue
        out = [{"title": a.get("title", "").strip(), "url": a.get("url", ""),
                "domain": a.get("domain", ""), "seendate": a.get("seendate", "")}
               for a in articles if a.get("title")]
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh)
        return out
    raise RuntimeError(f"GDELT: giving up on {query} {s}-{e}")


def fetch_range(query: str, start: dt.date, end: dt.date, window_days: int = 15,
                progress=None) -> List[Dict[str, str]]:
    seen, out = set(), []
    for a, b in windows(start, end, window_days):
        for item in fetch_window(query, a, b):
            key = item["title"].lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        if progress:
            progress(a, len(out))
    return out
