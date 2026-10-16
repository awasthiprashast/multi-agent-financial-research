"""
eval/sentiment_study.py — does FinBERT news sentiment line up with price moves?

    python -m eval.sentiment_study fetch      # headlines, Google News RSS (slow, cached)
    python -m eval.sentiment_study score      # FinBERT over every headline (cached)
    python -m eval.sentiment_study analyze    # correlation with returns -> results/
    python -m eval.sentiment_study all

Data: real. Headlines come from Google News RSS (date-windowed search; GDELT is
supported with --source gdelt but throttled this project's network), prices from
Yahoo Finance (yfinance), for eleven large-cap NSE equities over the last twelve
months. Nothing is simulated.

Method, so the number means what it says:
  * a headline is kept only if the company is named in the headline itself;
  * each headline is scored by FinBERT as P(positive) - P(negative);
  * a headline is assigned to the trading day on which it could first have been
    traded on: a headline on a non-trading day belongs to the next session (Google
    News gives dates, not reliable times, so intraday ordering is unknown and the
    same-day figure is a coincident association, not a prediction);
  * daily sentiment is the mean score of that day's headlines;
  * returns are *abnormal* returns (stock minus the Nifty 50 index) so a market-wide
    rally is not read as company sentiment;
  * we report same-day correlation (sentiment vs the move it coincides with) and
    next-day correlation (does it predict anything), with Pearson, Spearman and a
    permutation p-value that shuffles sentiment within each stock.
Weak or null results are reported as they come out.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")
RESULTS = os.path.join(ROOT, "results")
ARTICLES = os.path.join(CACHE, "sentiment_articles.json")
SCORES = os.path.join(CACHE, "sentiment_scores.json")

# ticker -> (GDELT query, headline aliases)
UNIVERSE: Dict[str, tuple] = {
    "TCS.NS": ('("Tata Consultancy Services" OR TCS)', ["tcs", "tata consultancy"]),
    "RELIANCE.NS": ('"Reliance Industries"', ["reliance"]),
    "INFY.NS": ("Infosys", ["infosys"]),
    "HDFCBANK.NS": ('"HDFC Bank"', ["hdfc bank"]),
    "ICICIBANK.NS": ('"ICICI Bank"', ["icici bank"]),
    "SBIN.NS": ('("State Bank of India" OR SBI)', ["state bank of india", "sbi"]),
    "ITC.NS": ('"ITC Ltd"', ["itc"]),
    "LT.NS": ('"Larsen & Toubro"', ["larsen", "l&t"]),
    "WIPRO.NS": ("Wipro", ["wipro"]),
    "BHARTIARTL.NS": ('"Bharti Airtel"', ["airtel"]),
    "HINDUNILVR.NS": ('"Hindustan Unilever"', ["hindustan unilever", "hul"]),
}
MARKET = "^NSEI"
IST = dt.timedelta(hours=5, minutes=30)
MODEL = "ProsusAI/finbert"


def _load(path: str, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return default


def _save(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)


def _period() -> tuple:
    end = dt.date.today() - dt.timedelta(days=1)
    return end - dt.timedelta(days=364), end


# ------------------------------------------------------------------- 1. fetch
def stage_fetch(window_days: int, source: str = "gnews") -> None:
    if source == "gdelt":
        from data.connectors.gdelt import fetch_range
    else:
        from data.connectors.gnews import fetch_range

    start, end = _period()
    store = _load(ARTICLES, {})
    for ticker, (query, aliases) in UNIVERSE.items():
        if ticker in store and store[ticker].get("window_days") == window_days \
                and store[ticker].get("end") == end.isoformat() \
                and store[ticker].get("source") == source:
            print(f"{ticker}: cached ({len(store[ticker]['items'])} headlines)")
            continue
        print(f"{ticker}: fetching {start} .. {end} from {source}", flush=True)
        items = fetch_range(query, start, end, window_days=window_days,
                            progress=lambda d, n: print(f"   {d} -> {n}", flush=True)
                            if d.day <= window_days and d.month % 3 == 0 else None)
        store[ticker] = {"window_days": window_days, "end": end.isoformat(),
                         "source": source, "items": items}
        _save(ARTICLES, store)
        print(f"{ticker}: {len(items)} unique headlines", flush=True)


def _relevant(title: str, aliases: List[str]) -> bool:
    low = title.lower()
    return any(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", low) for a in aliases)


# ------------------------------------------------------------------- 2. score
def stage_score() -> None:
    store = _load(ARTICLES, {})
    scores = _load(SCORES, {})
    todo = []
    for ticker, blob in store.items():
        aliases = UNIVERSE[ticker][1]
        for it in blob["items"]:
            if _relevant(it["title"], aliases):
                key = hashlib.sha1(it["title"].encode("utf-8")).hexdigest()
                if key not in scores:
                    todo.append((key, it["title"]))
    todo = list({k: t for k, t in todo}.items())
    print(f"scoring {len(todo)} new headlines with {MODEL}", flush=True)
    if not todo:
        return
    from transformers import pipeline

    pipe = pipeline("sentiment-analysis", model=MODEL, top_k=None,
                    truncation=True, max_length=128)
    B = 32
    for i in range(0, len(todo), B):
        batch = todo[i:i + B]
        out = pipe([t for _, t in batch], batch_size=B)
        for (key, _), res in zip(batch, out):
            p = {r["label"].lower(): float(r["score"]) for r in res}
            scores[key] = {"pos": p.get("positive", 0.0), "neg": p.get("negative", 0.0),
                           "neu": p.get("neutral", 0.0)}
        if (i // B) % 20 == 0:
            print(f"   {i + len(batch)}/{len(todo)}", flush=True)
            _save(SCORES, scores)
    _save(SCORES, scores)


# ----------------------------------------------------------------- 3. analyze
def _trading_day(seen: str, days: List[str]) -> Optional[str]:
    """First session on which a headline first seen at `seen` (UTC) is tradable."""
    try:
        t = dt.datetime.strptime(seen, "%Y%m%dT%H%M%SZ") + IST
    except ValueError:
        return None
    day = t.date()
    if t.hour * 60 + t.minute >= 15 * 60 + 30:
        day += dt.timedelta(days=1)
    iso = day.isoformat()
    i = int(np.searchsorted(days, iso))        # first trading day >= iso
    return days[i] if i < len(days) else None


def _prices(ticker: str):
    from data.connectors.live import fetch_prices

    p = fetch_prices(ticker)
    if not p:
        raise RuntimeError(f"no prices for {ticker}")
    return p["date"], np.array(p["close"], dtype=float)


def _corr(x, y):
    from scipy.stats import pearsonr, spearmanr

    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": int(len(x)), "pearson": None, "pearson_p": None,
                "spearman": None, "spearman_p": None}
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return {"n": int(len(x)), "pearson": round(float(pr), 4), "pearson_p": round(float(pp), 4),
            "spearman": round(float(sr), 4), "spearman_p": round(float(sp), 4)}


def _perm_p(groups: List[tuple], n_perm: int = 2000, seed: int = 11) -> float:
    """Permutation p-value for pooled Pearson r: shuffle sentiment within each stock."""
    rng = np.random.default_rng(seed)
    xs = np.concatenate([g[0] for g in groups])
    ys = np.concatenate([g[1] for g in groups])
    obs = abs(np.corrcoef(xs, ys)[0, 1])
    hits = 0
    for _ in range(n_perm):
        sh = np.concatenate([rng.permutation(g[0]) for g in groups])
        if abs(np.corrcoef(sh, ys)[0, 1]) >= obs:
            hits += 1
    return round((hits + 1) / (n_perm + 1), 4)


def stage_analyze() -> dict:
    store = _load(ARTICLES, {})
    scores = _load(SCORES, {})
    m_dates, m_close = _prices(MARKET)
    m_ret = dict(zip(m_dates[1:], m_close[1:] / m_close[:-1] - 1.0))

    fetched = relevant = scored = 0
    panel_rows, per_ticker, pooled = [], {}, {"same": [], "next": []}
    for ticker, blob in store.items():
        aliases = UNIVERSE[ticker][1]
        dates, close = _prices(ticker)
        ret = dict(zip(dates[1:], close[1:] / close[:-1] - 1.0))
        ar = {d: ret[d] - m_ret[d] for d in ret if d in m_ret}
        trade_days = sorted(ar)

        daily: Dict[str, List[float]] = {}
        n_rel = 0
        fetched += len(blob["items"])
        for it in blob["items"]:
            if not _relevant(it["title"], aliases):
                continue
            n_rel += 1
            sc = scores.get(hashlib.sha1(it["title"].encode("utf-8")).hexdigest())
            if sc is None:
                continue
            day = _trading_day(it["seendate"], trade_days)
            if day is None:
                continue
            scored += 1
            daily.setdefault(day, []).append(sc["pos"] - sc["neg"])
        relevant += n_rel

        s_same, y_same, s_next, y_next = [], [], [], []
        for i, d in enumerate(trade_days):
            if d not in daily:
                continue
            s = float(np.mean(daily[d]))
            panel_rows.append({"ticker": ticker, "date": d, "n_headlines": len(daily[d]),
                               "sentiment": round(s, 4), "abnormal_return": round(ar[d], 5)})
            s_same.append(s)
            y_same.append(ar[d])
            if i + 1 < len(trade_days):
                s_next.append(s)
                y_next.append(ar[trade_days[i + 1]])
        per_ticker[ticker] = {
            "headlines_fetched": len(blob["items"]), "headlines_company_named": n_rel,
            "days_with_news": len(s_same),
            "same_day": _corr(s_same, y_same), "next_day": _corr(s_next, y_next),
        }
        if len(s_same) >= 8:
            pooled["same"].append((np.array(s_same), np.array(y_same)))
            pooled["next"].append((np.array(s_next), np.array(y_next)))

    def pool(key):
        if not pooled[key]:
            return {}
        xs = np.concatenate([g[0] for g in pooled[key]])
        ys = np.concatenate([g[1] for g in pooled[key]])
        out = _corr(xs, ys)
        out["permutation_p"] = _perm_p(pooled[key])
        out["tickers"] = len(pooled[key])
        return out

    same, nxt = pool("same"), pool("next")
    start, end = _period()
    result = {
        "period": f"{start} to {end}",
        "model": MODEL,
        "universe": list(UNIVERSE),
        "headlines_fetched": fetched,
        "headlines_company_named": relevant,
        "headlines_scored_and_mapped_to_a_session": scored,
        "pooled_same_day_abnormal_return": same,
        "pooled_next_day_abnormal_return": nxt,
        "per_ticker": per_ticker,
        "method": ("headlines from Google News RSS filtered to those naming the company; "
                   "FinBERT P(pos)-P(neg); assigned to first tradable session; abnormal "
                   "return = stock minus Nifty 50"),
        "caveats": [
            "headlines, not article bodies",
            "Google News RSS returns at most ~100 items per query window; full windows "
            "are split, but coverage of very active names is still a sample",
            "publication dates only (no reliable time of day), so same-day correlation "
            "cannot separate cause from effect",
            "one year of data and eleven stocks: small-sample, low-power study",
            "correlation is not a trading signal; no costs, no out-of-sample test",
        ],
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "sentiment_study.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(RESULTS, "sentiment_panel.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["ticker", "date", "n_headlines",
                                           "sentiment", "abnormal_return"])
        w.writeheader()
        w.writerows(panel_rows)
    print(json.dumps({k: v for k, v in result.items() if k != "per_ticker"}, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["fetch", "score", "analyze", "all"])
    ap.add_argument("--window-days", type=int, default=7)
    ap.add_argument("--source", choices=["gnews", "gdelt"], default="gnews")
    args = ap.parse_args()
    sys.path.insert(0, ROOT)
    if args.stage in ("fetch", "all"):
        stage_fetch(args.window_days, args.source)
    if args.stage in ("score", "all"):
        stage_score()
    if args.stage in ("analyze", "all"):
        stage_analyze()


if __name__ == "__main__":
    main()
