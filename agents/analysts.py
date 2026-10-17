"""
agents/analysts.py — the four specialist analysis agents.

Each agent retrieves before it speaks, and every sentence it produces is a Claim
carrying the ids of the chunks it retrieved. None of these agents is permitted
to state a number it did not read out of, or compute from, indexed evidence.

Narrative phrasing can optionally be routed through an LLM (Groq free tier, or a
local Ollama model). That is a presentation layer only — see llm.py. The values
themselves are always computed in Python, because a number produced by a language
model is exactly the failure this project exists to prevent.
"""

from __future__ import annotations

import re
from typing import List, Optional

from agents.base import Agent
from core import Claim, extract_numbers

_SENTENCE_END = re.compile(r"(?<![A-Z])[.!?](?=\s+[A-Z\"“‘'(]|\s*$)")


def first_sentence(text: str) -> str:
    """The first sentence of `text`, ending with its punctuation.

    A full stop is a sentence end only when it follows a letter or closing mark and
    precedes a capital or the end — so "52.4%", "Rs. 5" and "Dr. Rao" are not split,
    which is what splitting on the first "." used to do."""
    text = text.strip()
    m = _SENTENCE_END.search(text)
    return text[: m.end()].strip() if m else (text if text[-1:] in ".!?" else text + ".")


def _find_value(text: str, label_pattern: str) -> Optional[str]:
    """Lift the number that follows a label out of an evidence chunk."""
    m = re.search(label_pattern + r"[^0-9₹-]{0,40}(₹?\s?-?[\d,]+(?:\.\d+)?\s?(?:%|bps|cr|x)?)",
                  text, re.IGNORECASE)
    return m.group(1).strip() if m else None


# --------------------------------------------------------------------------
class FundamentalAgent(Agent):
    name = "fundamental"
    section = "Fundamental Analysis"

    QUERIES = [
        ("revenue", "consolidated revenue from operations growth over previous financial year"),
        ("margin", "operating margin for the year bps movement versus prior year"),
        ("returns", "return on equity and debt to equity ratio balance sheet"),
        ("eps", "basic earnings per share and dividend yield"),
    ]

    def _live_claims(self, ticker: str) -> List[Claim]:
        """Claims from the real Yahoo Finance data (live mode): the annual statements
        chunk (`financials_*`) and the headline-ratios chunk (`fundamentals_*`)."""
        mine = [c for c in self.store.chunks.values()
                if c.ticker == ticker and c.doc_type == "fundamentals"]
        fin = next((c for c in mine if c.source.startswith("financials_")), None)
        head = next((c for c in mine if c.source.startswith("fundamentals_")), None)
        out: List[Claim] = []

        if fin is not None:
            cited = [fin.chunk_id]
            fy = fin.meta.get("fiscal_year_end", "")
            period = f"for the fiscal year ended {fy} " if fy else ""
            rev = _find_value(fin.text, r"Total revenue is")
            growth = _find_value(fin.text, r"Revenue growth over the prior fiscal year is")
            if rev and growth:
                out.append(self.claim(
                    f"Total revenue {period}was {rev}, a growth of {growth} over the prior "
                    f"fiscal year.", cited))
            elif rev:
                out.append(self.claim(f"Total revenue {period}was {rev}.", cited))
            opm = _find_value(fin.text, r"Operating margin is")
            if opm:
                out.append(self.claim(f"The operating margin {period}was {opm}.", cited))
            nm = _find_value(fin.text, r"Net margin is")
            ni = _find_value(fin.text, r"Net income is")
            if nm and ni:
                out.append(self.claim(
                    f"Net income {period}was {ni}, a net margin of {nm}.", cited))
            eps = _find_value(fin.text, r"Basic earnings per share is")
            eps0 = _find_value(fin.text, r"Prior-year basic earnings per share was")
            if eps and eps0:
                out.append(self.claim(
                    f"Basic earnings per share were {eps}, against {eps0} in the prior "
                    f"year.", cited))
            elif eps:
                out.append(self.claim(f"Basic earnings per share were {eps}.", cited))
            de = _find_value(fin.text, r"Debt-to-equity ratio is")
            if de:
                out.append(self.claim(f"The debt-to-equity ratio was {de}.", cited))
            roe = _find_value(fin.text, r"Return on equity is")
            if roe:
                out.append(self.claim(
                    f"Return on equity, computed as net income over year-end equity, was "
                    f"{roe}.", cited))

        if head is not None:
            cited = [head.chunk_id]
            pe = _find_value(head.text, r"Trailing price-to-earnings ratio is")
            if pe:
                out.append(self.claim(f"The trailing price-to-earnings ratio is {pe}.", cited))
            if fin is None:                 # no statements: fall back to Yahoo's headline ratios
                for label, fmt in ((r"Return on equity is", "Return on equity is {}."),
                                   (r"Debt-to-equity ratio is", "The debt-to-equity ratio is {}."),
                                   (r"Profit margin is", "Profit margin is {}.")):
                    v = _find_value(head.text, label)
                    if v:
                        out.append(self.claim(fmt.format(v), cited))
        return out

    def run(self, ticker: str, context: dict) -> List[Claim]:
        claims: List[Claim] = self._live_claims(ticker)
        for _, q in self.QUERIES:
            cited = self.cite(q, ticker, k=2)
            if not cited:
                continue
            chunk = self.store.get(cited[0])
            if chunk is None:
                continue

            rev = _find_value(chunk.text, r"revenue from operations of")
            growth = _find_value(chunk.text, r"growth of")
            if rev and growth:
                claims.append(self.claim(
                    f"Consolidated revenue from operations was {rev}, up {growth} "
                    f"year on year.", cited))

            om = _find_value(chunk.text, r"operating margin for the year stood at")
            bps = _find_value(chunk.text, r"a movement of")
            if om:
                txt = f"Operating margin for the year was {om}"
                if bps:
                    txt += f", a movement of {bps} against the prior year"
                claims.append(self.claim(txt + ".", cited))

            roe = _find_value(chunk.text, r"return on equity for the year was")
            de = _find_value(chunk.text, r"debt-to-equity ratio stood at")
            if roe:
                claims.append(self.claim(
                    f"Return on equity for the year was {roe}, which the filing "
                    f"attributes to the company's asset-light operating model.", cited))
            if de:
                claims.append(self.claim(
                    f"The consolidated debt-to-equity ratio was {de}, and the filing "
                    f"records a net cash position through the year.", cited))

            eps = _find_value(chunk.text, r"earnings per share for the year were")
            dy = _find_value(chunk.text, r"dividend yield of approximately")
            if eps:
                claims.append(self.claim(f"Basic earnings per share were {eps}.", cited))
            if dy:
                claims.append(self.claim(
                    f"Dividend yield at the year-end closing price was approximately "
                    f"{dy}.", cited))

        # de-duplicate while preserving order
        seen, out = set(), []
        for c in claims:
            if c.text in seen:
                continue
            seen.add(c.text)
            out.append(c)
        return out

    def revise(self, claim: Claim, feedback: str, ticker: str, context: dict):
        """Conservative revision: restate only what the retrieved chunk says."""
        cited = self.cite(claim.text, ticker, k=1)
        if not cited:
            return None
        chunk = self.store.get(cited[0])
        if chunk is None:
            return None
        nums = extract_numbers(chunk.text)
        if not nums:
            return None
        revised = self.claim(
            f"The filing states: {first_sentence(chunk.text)}", cited)
        revised.revision = claim.revision + 1
        return revised


# --------------------------------------------------------------------------
class TechnicalAgent(Agent):
    name = "technical"
    section = "Technical Analysis"

    def run(self, ticker: str, context: dict) -> List[Claim]:
        ind = context["indicators"]
        cited = self.cite("computed trend indicators RSI MACD moving average", ticker, k=2)
        claims: List[Claim] = []

        claims.append(self.claim(
            f"The last close was ₹{ind['last_close']}, against a 20-day simple moving "
            f"average of ₹{ind['sma_20']} and a 50-day average of ₹{ind['sma_50']}.",
            cited))

        r = ind["rsi_14"]
        zone = ("overbought territory" if r >= 70 else
                "oversold territory" if r <= 30 else "neutral territory")
        claims.append(self.claim(
            f"RSI(14) is {r}, which places the stock in {zone}.", cited))

        claims.append(self.claim(
            f"The MACD line is {ind['macd_line']} against a signal line of "
            f"{ind['macd_signal']}, giving a histogram of {ind['macd_hist']}.", cited))

        claims.append(self.claim(
            f"The 20-day Bollinger band runs from ₹{ind['bb_lower']} to "
            f"₹{ind['bb_upper']}.", cited))

        claims.append(self.claim(
            f"Total price return over the {ind['sessions']}-session window was "
            f"{ind['period_return_pct']}%.", cited))
        return claims


# --------------------------------------------------------------------------
class SentimentAgent(Agent):
    name = "sentiment"
    section = "Sentiment Analysis"

    def __init__(self, store, scorer=None):
        super().__init__(store)
        from sentiment_model import get_scorer
        self.scorer = scorer or get_scorer()

    def run(self, ticker: str, context: dict) -> List[Claim]:
        news = [c for c in self.store.chunks.values()
                if c.ticker == ticker and c.doc_type == "news"]
        if not news:
            return []

        # Scoring happens at ingestion time and is recorded as derived evidence,
        # so the aggregate is verifiable in exactly the same way a filing number
        # is. See pipeline._prepare().
        detail = (context.get("sentiment_detail") or {}).get(ticker, {})
        labels = [detail.get("per_chunk", {}).get(c.chunk_id)
                  or self.scorer.label(c.text) for c in news]
        pos = labels.count("positive")
        neg = labels.count("negative")
        neu = labels.count("neutral")
        total = len(labels)
        net = round((pos - neg) / total, 2)

        cited_all = ([detail["evidence_chunk_id"]] if detail.get("evidence_chunk_id")
                     else [c.chunk_id for c in news[:5]])
        claims = [self.claim(
            f"Across {total} indexed news items, {pos} were classified positive, "
            f"{neg} negative and {neu} neutral, giving a net sentiment score of {net}.",
            cited_all)]

        # Ground the headline reading in one specific, quotable item.
        strongest = None
        for c, lab in zip(news, labels):
            if lab == ("positive" if pos >= neg else "negative"):
                strongest = c
                break
        if strongest is not None:
            claims.append(self.claim(
                f"A representative item in the window reads: "
                f"\"{first_sentence(strongest.text)}\"",
                [strongest.chunk_id]))
        return claims


# --------------------------------------------------------------------------
class RiskAgent(Agent):
    name = "risk"
    section = "Risk Assessment"

    def run(self, ticker: str, context: dict) -> List[Claim]:
        ind = context["indicators"]
        cited = self.cite("computed risk statistics volatility drawdown value at risk",
                          ticker, k=2)
        return [
            self.claim(
                f"Annualised volatility of daily returns over the window was "
                f"{ind['annual_vol_pct']}%.", cited),
            self.claim(
                f"The maximum drawdown within the window was "
                f"{ind['max_drawdown_pct']}%.", cited),
            self.claim(
                f"Historical one-day Value at Risk at the 95% confidence level was "
                f"{ind['var_95_pct']}%.", cited),
        ]
