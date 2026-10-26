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


#: the hand-curated companies (the 12 above); anything else is registered at run time
CURATED_TICKERS: List[str] = [c.ticker for c in COMPANIES]


def supported_tickers(offline: bool) -> List[str]:
    """Offline: the 3 snapshot companies. Live: the curated 12 plus every company
    registered so far with `register_company`."""
    return list(SNAPSHOT_TICKERS) if offline else [c.ticker for c in COMPANIES]


class UnknownCompanyError(ValueError):
    """The query names no company this deployment can analyse."""


# ------------------------------------------------------------ any listed company
# Companies outside the curated 12 are found through Yahoo Finance's search and
# registered here, so that everything keyed on this module (news filtering, the
# verifier's wrong-company check, display names) works for them too.
import logging  # noqa: E402
import threading  # noqa: E402
from functools import lru_cache  # noqa: E402

log = logging.getLogger(__name__)
_register_lock = threading.Lock()

_SUFFIXES = ("limited", "ltd", "corporation", "corp", "incorporated", "inc", "company", "co",
             "plc", "pvt", "private")
# Yahoo's search matches the current name only; a few well-known renames
_RENAMED = {"zomato": "eternal", "tata motors": "tata motors", "tatamotors": "tata motors"}
_GENERIC_ALIAS = frozenset({"india", "indian", "bank", "group", "power", "steel", "oil",
                            "gas", "energy", "finance", "capital", "industries", "global"})


@dataclass(frozen=True)
class Candidate:
    ticker: str
    name: str
    exchange: str


def _clean_name(name: str) -> str:
    """"ASIAN PAINTS LTD." -> "Asian Paints"."""
    words = re.sub(r"[.,]", " ", name).split()
    while words and words[-1].lower() in _SUFFIXES:
        words.pop()
    out = " ".join(words)
    if out.isupper():
        out = " ".join(w if len(w) <= 3 else w.title() for w in out.split())
    return out or name


def _aliases_for(ticker: str, name: str) -> Tuple[str, ...]:
    """Lower-case surface forms a headline or sentence might use for the company."""
    base = _clean_name(name).lower()
    stem = ticker.split(".")[0].lower()
    forms = [base]
    if len(stem) >= 4 and stem.isalnum() and stem != base:
        forms.append(stem)                          # "paytm", "asianpaint"
    forms =[f for f in forms if f and f not in _GENERIC_ALIAS]
    return tuple(sorted(set(forms), key=len, reverse=True))


def register_company(ticker: str, name: str) -> Company:
    """Make `ticker` a known company. Idempotent; safe to call from several threads."""
    with _register_lock:
        existing = BY_TICKER.get(ticker)
        if existing is not None:
            return existing
        display = _clean_name(name)
        aliases = _aliases_for(ticker, name)
        co = Company(ticker, display, aliases, f'"{display}"')
        COMPANIES.append(co)
        BY_TICKER[ticker] = co
        NAME[ticker] = display
        for a in aliases:
            TICKER_ALIASES.setdefault(a, ticker)
            _PATTERNS.append((_alias_pattern(a), ticker))
        log.info("registered %s as %r (aliases %s)", ticker, display, list(aliases))
        return co


def _main_listing(symbol: str) -> bool:
    """NSE/BSE ordinary shares: not the odd series Yahoo also lists (-BL, -BE, -SM ...)."""
    return symbol.endswith((".NS", ".BO")) and "-" not in symbol.split(".")[0]


@lru_cache(maxsize=256)
def search_companies(query: str, limit: int = 5) -> Tuple[Candidate, ...]:
    """NSE/BSE companies matching `query` (a name or a ticker), best first. Network call
    to Yahoo Finance's search; returns () if it is unreachable or finds nothing."""
    q = query.strip()
    if len(q) < 2:
        return ()
    q = _RENAMED.get(q.lower(), q)
    try:
        import requests

        r = requests.get("https://query2.finance.yahoo.com/v1/finance/search",
                         params={"q": q, "quotesCount": 12, "newsCount": 0},
                         headers={"User-Agent": "Mozilla/5.0 (finresearch-assistant)"}, timeout=10)
        r.raise_for_status()
        quotes = r.json().get("quotes", [])
    except Exception as exc:
        log.warning("company search for %r failed: %s: %s", q, type(exc).__name__, exc)
        return ()
    found: List[Candidate] = []
    seen_names = set()
    for x in sorted(quotes, key=lambda x: not str(x.get("symbol", "")).endswith(".NS")):
        sym = str(x.get("symbol", ""))
        if x.get("quoteType") != "EQUITY" or not _main_listing(sym):
            continue
        name = x.get("longname") or x.get("shortname") or sym
        key = _clean_name(name).lower()
        if key in seen_names:                       # the same company on both exchanges
            continue
        seen_names.add(key)
        found.append(Candidate(sym, _clean_name(name),
                               "NSE" if sym.endswith(".NS") else "BSE"))
        if len(found) >= limit:
            break
    return tuple(found)


def ensure_registered(ticker: str) -> Optional[Company]:
    """The Company for `ticker`, looking it up and registering it if it is a real NSE/BSE
    symbol we have not seen. None if it cannot be found."""
    co = BY_TICKER.get(ticker)
    if co is not None:
        return co
    if not _main_listing(ticker):
        return None
    for cand in search_companies(ticker.split(".")[0]):
        if cand.ticker == ticker:
            return register_company(cand.ticker, cand.name)
    return None
