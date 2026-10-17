"""
data/universe.py — the companies the system knows about. One source of truth.

Everything that needs to recognise a company — the router (ticker resolution), the
verifier (entity binding), the corpus (display names), the API (which tickers are
accepted) — reads this list. Adding a company means adding one `Company` here.

Two kinds of support:
  * live      any company in `COMPANIES`: real prices, real fundamentals, real news;
  * snapshot  the three companies with a synthetic offline corpus (`SNAPSHOT_TICKERS`),
              used for the controlled benchmark. Offline mode only serves these.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Company:
    ticker: str                    # Yahoo Finance symbol
    name: str                      # display name
    aliases: Tuple[str, ...]       # lower-case surface forms, longest first
    news_query: str                # Google News search expression


COMPANIES: List[Company] = [
    Company("TCS.NS", "Tata Consultancy Services",
            ("tata consultancy services", "tata consultancy", "tcs"),
            '("Tata Consultancy Services" OR TCS)'),
    Company("RELIANCE.NS", "Reliance Industries",
            ("reliance industries", "reliance", "ril"), '"Reliance Industries"'),
    Company("INFY.NS", "Infosys", ("infosys", "infy"), "Infosys"),
    Company("HDFCBANK.NS", "HDFC Bank", ("hdfc bank", "hdfcbank"), '"HDFC Bank"'),
    Company("ICICIBANK.NS", "ICICI Bank", ("icici bank", "icicibank"), '"ICICI Bank"'),
    Company("SBIN.NS", "State Bank of India",
            ("state bank of india", "sbin", "sbi"), '("State Bank of India" OR SBI)'),
    Company("ITC.NS", "ITC Limited", ("itc limited", "itc ltd", "itc"), '"ITC Ltd"'),
    Company("LT.NS", "Larsen & Toubro",
            ("larsen & toubro", "larsen and toubro", "larsen", "l&t"), '"Larsen & Toubro"'),
    Company("WIPRO.NS", "Wipro", ("wipro",), "Wipro"),
    Company("BHARTIARTL.NS", "Bharti Airtel", ("bharti airtel", "airtel"), '"Bharti Airtel"'),
    Company("HINDUNILVR.NS", "Hindustan Unilever",
            ("hindustan unilever", "hindunilvr", "hul"), '"Hindustan Unilever"'),
    Company("ADANIGREEN.NS", "Adani Green Energy",
            ("adani green energy", "adani green", "adani"), '"Adani Green Energy"'),
]

#: the companies with a synthetic offline corpus (see data/corpus.GROUND_TRUTH)
SNAPSHOT_TICKERS: List[str] = ["TCS.NS", "RELIANCE.NS", "INFY.NS"]

BY_TICKER: Dict[str, Company] = {c.ticker: c for c in COMPANIES}
NAME: Dict[str, str] = {c.ticker: c.name for c in COMPANIES}
TICKER_ALIASES: Dict[str, str] = {a: c.ticker for c in COMPANIES for a in c.aliases}


def _alias_pattern(alias: str) -> "re.Pattern[str]":
    return re.compile(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])")


_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (_alias_pattern(a), c.ticker) for c in COMPANIES for a in c.aliases]


def find_tickers(text: str) -> List[str]:
    """Tickers mentioned in `text`, in order of first mention, without duplicates."""
    low = text.lower()
    hits: List[Tuple[int, str]] = []
    for pat, ticker in _PATTERNS:
        m = pat.search(low)
        if m:
            hits.append((m.start(), ticker))
    out: List[str] = []
    for _, t in sorted(hits):
        if t not in out:
            out.append(t)
    return out


def mentions(text: str, ticker: str) -> bool:
    """Does `text` name this company (whole-word alias match)?"""
    co = BY_TICKER.get(ticker)
    if co is None:
        return False
    low = text.lower()
    return any(_alias_pattern(a).search(low) for a in co.aliases)


def resolve_ticker(query: str) -> Optional[str]:
    """The first company named in `query`, or None. Never guesses a default."""
    found = find_tickers(query)
    return found[0] if found else None


def supported_tickers(offline: bool) -> List[str]:
    return list(SNAPSHOT_TICKERS) if offline else [c.ticker for c in COMPANIES]


class UnknownCompanyError(ValueError):
    """The query names no company this deployment can analyse."""
