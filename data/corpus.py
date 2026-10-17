"""
data/corpus.py — corpus assembly.

Two modes, one interface:

  LIVE      the connectors in data/connectors/live.py pull real filings, prices
            and RSS news. Used on any machine with network access.

  SNAPSHOT  a frozen, seeded, fully self-contained evaluation corpus shipped with
            the repo. Every document carries provenance="synthetic-demo" and the
            true value of every quantity is recorded in GROUND_TRUTH.

The snapshot exists for a specific methodological reason. Measuring a
verification system requires knowing, for every claim, whether it is actually
supported — which means a corpus with annotated ground truth. Live market data
gives you no such labels. The snapshot is therefore the controlled benchmark on
which grounding rate and hallucination catch rate are measured; the live path is
what the deployed system runs on. Both go through exactly the same agents.
"""

from __future__ import annotations

import logging
import math
import random
import zlib
from typing import Dict, List, Tuple

from core import Chunk
from data.universe import BY_TICKER, NAME, mentions

log = logging.getLogger(__name__)

SEED = 20260902

# --------------------------------------------------------------------------
# Ground truth for the snapshot companies. Every number that appears in a
# snapshot document is derived from this table, so the evaluation harness knows
# what "supported" means.
# --------------------------------------------------------------------------

GROUND_TRUTH: Dict[str, Dict[str, float]] = {
    "TCS.NS": {
        "revenue_cr": 240893.0, "revenue_growth_pct": 6.8, "pe": 26.4,
        "roe_pct": 51.2, "debt_to_equity": 0.09, "operating_margin_pct": 24.3,
        "margin_change_bps": -70.0, "eps": 134.2, "dividend_yield_pct": 1.6,
        "start_price": 3480.0, "annual_vol_pct": 19.4,
    },
    "RELIANCE.NS": {
        "revenue_cr": 901064.0, "revenue_growth_pct": 7.1, "pe": 24.8,
        "roe_pct": 9.3, "debt_to_equity": 0.44, "operating_margin_pct": 16.9,
        "margin_change_bps": 40.0, "eps": 102.9, "dividend_yield_pct": 0.4,
        "start_price": 1420.0, "annual_vol_pct": 24.1,
    },
    "INFY.NS": {
        "revenue_cr": 158381.0, "revenue_growth_pct": 4.7, "pe": 23.1,
        "roe_pct": 31.8, "debt_to_equity": 0.11, "operating_margin_pct": 21.1,
        "margin_change_bps": 30.0, "eps": 63.4, "dividend_yield_pct": 2.4,
        "start_price": 1560.0, "annual_vol_pct": 22.7,
    },
}

COMPANY_NAME = NAME                    # display names; single list in data/universe.py


class DataUnavailableError(RuntimeError):
    """No live data could be fetched and there is no offline snapshot to fall back on."""


# --------------------------------------------------------------------------
# Document templates
# --------------------------------------------------------------------------

def _filing_paragraphs(ticker: str) -> List[Tuple[str, str]]:
    g = GROUND_TRUTH[ticker]
    name = COMPANY_NAME[ticker]
    return [
        (f"AR2026_{ticker}_MDA_p31",
         f"For the financial year under review, {name} reported consolidated revenue "
         f"from operations of ₹{g['revenue_cr']:,.0f} cr, representing growth of "
         f"{g['revenue_growth_pct']}% over the previous financial year. Growth was "
         f"broad-based across geographies with the exception of continental Europe, "
         f"where client decision cycles remained extended through the second half."),
        (f"AR2026_{ticker}_MDA_p32",
         f"Operating margin for the year stood at {g['operating_margin_pct']}%, a "
         f"movement of {g['margin_change_bps']:.0f} bps versus the prior year. The "
         f"movement reflects wage revision effects and a higher proportion of "
         f"onsite effort, partially offset by improved utilisation and pyramid "
         f"rationalisation."),
        (f"AR2026_{ticker}_FIN_p58",
         f"Return on equity for the year was {g['roe_pct']}%. The consolidated "
         f"debt-to-equity ratio stood at {g['debt_to_equity']} as at the balance "
         f"sheet date, and the company continued to operate with a net cash "
         f"position through the year."),
        (f"AR2026_{ticker}_FIN_p61",
         f"Basic earnings per share for the year were ₹{g['eps']}. The Board has "
         f"recommended a final dividend which, together with interim dividends "
         f"declared during the year, corresponds to a dividend yield of "
         f"approximately {g['dividend_yield_pct']}% at the year-end closing price."),
        (f"Q4FY26_{ticker}_RESULT_p2",
         f"The Board of Directors of {name} approved the audited financial results "
         f"for the quarter and year ended 31 March 2026. Order book remained healthy "
         f"and the total contract value signed during the quarter was the highest in "
         f"eight quarters, though the company cautioned that conversion timelines "
         f"remain sensitive to the macro environment."),
        (f"Q4FY26_{ticker}_RESULT_p4",
         "Attrition on a trailing twelve-month basis moderated further during the "
         "quarter. Headcount declined marginally on a sequential basis as the "
         "company continued to prioritise utilisation over gross additions."),
    ]


_NEWS_POS = [
    "{name} shares advanced after the company reported a stronger-than-expected order "
    "book, with analysts pointing to improving deal conversion in the BFSI segment.",
    "Brokerages raised their target price on {name}, citing margin resilience and a "
    "healthy net cash position on the balance sheet.",
    "{name} announced a large multi-year engagement with a European client, its "
    "biggest signing in the region for eight quarters.",
]

_NEWS_NEG = [
    "{name} slipped in trade after management flagged extended client decision cycles "
    "in continental Europe and a softer discretionary spending environment.",
    "Concerns over wage revision pressure weighed on {name}, with several analysts "
    "trimming near-term margin assumptions.",
    "{name} faced questions over slowing headcount addition, which some analysts read "
    "as a signal of weaker forward demand.",
]

_NEWS_NEU = [
    "{name} will announce its quarterly results later this month, according to an "
    "exchange filing dated this week.",
    "{name} said its annual general meeting will be held in the last week of the "
    "quarter, as per the notice filed with the exchanges.",
    "Trading volumes in {name} were broadly in line with the twenty-day average "
    "during the session.",
]


def _news_items(ticker: str, rng: random.Random) -> List[Tuple[str, str, str]]:
    """Returns (chunk_source, text, gold_sentiment_label)."""
    name = COMPANY_NAME[ticker]
    out: List[Tuple[str, str, str]] = []
    pools = [(_NEWS_POS, "positive"), (_NEWS_NEG, "negative"), (_NEWS_NEU, "neutral")]
    idx = 0
    for _ in range(4):
        for pool, label in pools:
            tmpl = pool[idx % len(pool)]
            out.append((f"news_{ticker}_{idx:03d}", tmpl.format(name=name), label))
            idx += 1
    rng.shuffle(out)
    return out


def _price_series(ticker: str, n: int = 252) -> Dict[str, list]:
    """Seeded geometric Brownian motion at the company's ground-truth volatility."""
    g = GROUND_TRUTH[ticker]
    # zlib.crc32, not hash(): str hashing is salted per process, which made the
    # "seeded" series differ from run to run.
    rng = random.Random(SEED + zlib.crc32(ticker.encode("utf-8")) % 10_000)
    daily_sigma = (g["annual_vol_pct"] / 100.0) / math.sqrt(252)
    mu = 0.08 / 252
    price = g["start_price"]
    closes, highs, lows, opens, vols = [], [], [], [], []
    for _ in range(n):
        shock = rng.gauss(mu - 0.5 * daily_sigma ** 2, daily_sigma)
        o = price
        price = price * math.exp(shock)
        h = max(o, price) * (1 + abs(rng.gauss(0, daily_sigma / 3)))
        lo = min(o, price) * (1 - abs(rng.gauss(0, daily_sigma / 3)))
        opens.append(round(o, 2))
        closes.append(round(price, 2))
        highs.append(round(h, 2))
        lows.append(round(lo, 2))
        vols.append(float(int(rng.gauss(4.2e6, 8e5))))
    dates = [f"2026-{1 + (i // 21) % 12:02d}-{1 + i % 28:02d}" for i in range(n)]
    return {
        "date": dates, "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols, "provenance": "synthetic-demo",
    }


def statements_chunk(ticker: str, stmt: dict):
    """One evidence chunk of annual financials, every ratio computed here in Python.

    Sentences are short and carry one label each ("Operating margin is 25.1%.") so
    the verifier's label binding can match a claim to exactly the right number."""
    if not stmt or (stmt.get("currency") or "INR") != "INR" or not stmt.get("years"):
        return None
    y0 = stmt["years"][0]
    y1 = stmt["years"][1] if len(stmt["years"]) > 1 else {}
    cr = lambda v: v / 1e7                                    # rupees -> crore

    s: List[str] = []
    if y0.get("revenue"):
        s.append(f"Total revenue is ₹{cr(y0['revenue']):,.0f} cr.")
        if y1.get("revenue"):
            g = (y0["revenue"] / y1["revenue"] - 1) * 100
            s.append(f"Revenue growth over the prior fiscal year is {g:.1f}%.")
            s.append(f"Prior-year total revenue was ₹{cr(y1['revenue']):,.0f} cr.")
        if y0.get("operating_income"):
            s.append(f"Operating margin is {y0['operating_income'] / y0['revenue'] * 100:.1f}%.")
        if y0.get("net_income"):
            s.append(f"Net margin is {y0['net_income'] / y0['revenue'] * 100:.1f}%.")
    if y0.get("net_income"):
        s.append(f"Net income is ₹{cr(y0['net_income']):,.0f} cr.")
    if y0.get("eps"):
        s.append(f"Basic earnings per share is ₹{y0['eps']:.2f}.")
        if y1.get("eps"):
            s.append(f"Prior-year basic earnings per share was ₹{y1['eps']:.2f}.")
    # Banks and lenders report no operating income, and their "debt" includes deposits,
    # so a debt-to-equity ratio would mislead: only state it for non-financial companies.
    if y0.get("total_debt") is not None and y0.get("equity") and y0.get("operating_income"):
        s.append(f"Debt-to-equity ratio is {y0['total_debt'] / y0['equity']:.2f}.")
    if y0.get("net_income") and y0.get("equity"):
        s.append(f"Return on equity is {y0['net_income'] / y0['equity'] * 100:.1f}%.")
    if not s:
        return None
    name = COMPANY_NAME.get(ticker, ticker)
    text = (f"Annual financials for {name} from Yahoo Finance statements, fiscal year ended "
            f"{y0['end']}. " + " ".join(s) +
            " Return on equity is net income divided by year-end equity; debt-to-equity is "
            "total debt divided by equity.")
    src = f"financials_{ticker}"
    return Chunk(chunk_id=Chunk.make_id(src, text), text=text, source=src,
                 doc_type="fundamentals", ticker=ticker, published=y0["end"],
                 meta={"provenance": "yfinance-statements", "fiscal_year_end": y0["end"]})


def _live_fundamentals_chunk(ticker: str):
    """Real headline fundamentals (Yahoo Finance) as one evidence chunk, or None."""
    try:
        from data.connectors.live import fetch_fundamentals

        f = fetch_fundamentals(ticker)
    except Exception:
        return None
    if not f:
        return None
    parts = []

    def add(label: str, value, fmt: str) -> None:
        if value is not None and value == value:
            parts.append(f"{label} {fmt.format(value)}.")

    add("Trailing price-to-earnings ratio is", f.get("pe"), "{:.1f}")
    add("Return on equity is", f.get("roe"), "{:.1f}%")
    add("Debt-to-equity ratio is", f.get("debt_to_equity"), "{:.1f}%")
    add("Profit margin is", f.get("profit_margin"), "{:.1f}%")
    if f.get("market_cap"):
        parts.append(f"Market capitalisation is {f['market_cap'] / 1e7:,.0f} cr.")
    if not parts:
        return None
    name = COMPANY_NAME.get(ticker, ticker)
    text = (f"Headline fundamentals for {name} from Yahoo Finance as of "
            f"{f.get('retrieved', '')}. " + " ".join(parts))
    src = f"fundamentals_{ticker}"
    return Chunk(chunk_id=Chunk.make_id(src, text), text=text, source=src,
                 doc_type="fundamentals", ticker=ticker,
                 published=f.get("retrieved", ""),
                 meta={"provenance": f.get("provenance", "yfinance.info")})


LIVE_NEWS_DAYS = 30
LIVE_NEWS_MAX = 40


def _live_news_chunks(ticker: str, rss_fallback) -> List[Chunk]:
    """Recent headlines that name the company: Google News search, falling back to
    the plain RSS feeds if that is unreachable. Headlines only (no article bodies)."""
    import datetime as dt

    co = BY_TICKER.get(ticker)
    items: List[dict] = []
    try:
        from data.connectors.gnews import fetch_range

        end = dt.date.today()
        start = end - dt.timedelta(days=LIVE_NEWS_DAYS)
        raw = fetch_range(co.news_query if co else COMPANY_NAME.get(ticker, ticker),
                          start, end, window_days=15)
        named = [i for i in raw if mentions(i["title"], ticker)]
        named.sort(key=lambda i: i["seendate"], reverse=True)
        for i in named[:LIVE_NEWS_MAX]:
            d = i["seendate"]
            items.append({"text": i["title"], "source": i.get("source") or "google-news",
                          "url": i.get("url", ""),
                          "published": f"{d[0:4]}-{d[4:6]}-{d[6:8]}",
                          "provenance": "google-news-rss"})
    except Exception as exc:
        log.warning("Google News unavailable for %s (%s); trying plain RSS", ticker, exc)

    if not items:
        try:
            for it in rss_fallback(COMPANY_NAME.get(ticker, ticker)) or []:
                items.append({"text": f"{it['title']}. {it['summary']}",
                              "source": it.get("link", "rss"), "url": it.get("link", ""),
                              "published": it.get("published", ""), "provenance": "rss-live"})
        except Exception as exc:
            log.warning("RSS fallback failed for %s (%s)", ticker, exc)

    chunks: List[Chunk] = []
    for n, it in enumerate(items):
        src = f"news_{ticker}_{n:03d}"
        chunks.append(Chunk(
            chunk_id=Chunk.make_id(src, it["text"]), text=it["text"], source=src,
            doc_type="news", ticker=ticker, published=it["published"],
            meta={"provenance": it["provenance"], "publisher": it["source"], "url": it["url"]}))
    return chunks


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def load_corpus(ticker: str, prefer_live: bool = True):
    """Return (chunks, prices, news_gold, mode) for one ticker."""
    rng = random.Random(SEED)
    chunks: List[Chunk] = []
    mode = "snapshot"
    prices = None

    if prefer_live:
        try:
            from data.connectors.live import fetch_news, fetch_prices

            prices = fetch_prices(ticker)
            if prices:
                mode = "live"
                chunks.extend(_live_news_chunks(ticker, fetch_news))
        except Exception as exc:
            log.warning("live data for %s failed (%s)", ticker, exc)
            prices = None

    if prices is None:
        if ticker not in GROUND_TRUTH:
            raise DataUnavailableError(
                f"no live prices for {ticker} and no offline snapshot exists for it "
                f"(snapshot companies: {', '.join(GROUND_TRUTH)})")
        if prefer_live:
            log.warning("falling back to the SYNTHETIC snapshot for %s; the report will "
                        "say corpus=snapshot", ticker)
        prices = _price_series(ticker)

    news_gold: Dict[str, str] = {}

    if mode == "snapshot":
        # Synthetic filings, generated from GROUND_TRUTH. Never mixed with live data:
        # a report on real prices must not quote an invented annual report.
        for src, text in _filing_paragraphs(ticker):
            cid = Chunk.make_id(src, text)
            chunks.append(Chunk(chunk_id=cid, text=text, source=src,
                                doc_type="filing", ticker=ticker,
                                published="2026-04-18",
                                meta={"provenance": "synthetic-demo"}))
    else:
        fund_chunk = _live_fundamentals_chunk(ticker)
        if fund_chunk is not None:
            chunks.append(fund_chunk)
        try:
            from data.connectors.live import fetch_statements
            stmt_chunk = statements_chunk(ticker, fetch_statements(ticker))
        except Exception as exc:
            log.warning("statements for %s unavailable (%s)", ticker, exc)
            stmt_chunk = None
        if stmt_chunk is not None:
            chunks.append(stmt_chunk)

    if mode == "snapshot":
        for src, text, label in _news_items(ticker, rng):
            cid = Chunk.make_id(src, text)
            chunks.append(Chunk(chunk_id=cid, text=text, source=src,
                                doc_type="news", ticker=ticker,
                                published="2026-08-2%d" % (rng.randint(0, 8)),
                                meta={"provenance": "synthetic-demo", "gold_sentiment": label}))
            news_gold[cid] = label

    # Market data is itself evidence — index a textual summary of it so price
    # claims are groundable in exactly the same way filing claims are.
    closes = prices["close"]
    summary = (
        f"Market data for {COMPANY_NAME.get(ticker, ticker)}: the closing price on the "
        f"last session in the window was ₹{closes[-1]:.2f}, against ₹{closes[0]:.2f} at "
        f"the start of the window, over {len(closes)} trading sessions. The highest "
        f"close in the window was ₹{max(closes):.2f} and the lowest was ₹{min(closes):.2f}."
    )
    cid = Chunk.make_id(f"marketdata_{ticker}", summary)
    chunks.append(Chunk(chunk_id=cid, text=summary, source=f"marketdata_{ticker}",
                        doc_type="market_data", ticker=ticker, published="2026-08-31",
                        meta={"provenance": prices.get("provenance", "snapshot")}))

    return chunks, prices, news_gold, mode


def all_tickers() -> List[str]:
    return list(GROUND_TRUTH.keys())
