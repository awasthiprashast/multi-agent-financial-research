"""
agents/synthesis.py — the "Overall view" section.

The four specialist agents each speak in isolation. This agent reads their *inputs*
(the indicators, the sentiment record, the reported financials) and states where the
signals agree or diverge. It is deterministic: no LLM, so nothing here can be invented.

Every sentence is a Claim like any other. Each is built from facts the Verification
Agent can check on its own — price vs moving averages, RSI zone, MACD vs signal,
volatility and drawdown, net sentiment score, revenue growth, period return — so the
synthesis is verified, not trusted. The agree/diverge wording is a fixed function of
those facts. It never recommends anything.
"""

from __future__ import annotations

from typing import List, Optional, Set

from agents.analysts import _find_value
from agents.base import Agent
from core import Claim

SENTIMENT_BAND = 0.10            # |net score| below this reads as neutral


def _trend(ind: dict) -> str:
    c, s20, s50 = ind["last_close"], ind["sma_20"], ind["sma_50"]
    if c > s20 and c > s50:
        return "above"
    if c < s20 and c < s50:
        return "below"
    return "mixed"


def _tone(net: float) -> str:
    return "positive" if net > SENTIMENT_BAND else "negative" if net < -SENTIMENT_BAND else "neutral"


class SynthesisAgent(Agent):
    name = "synthesis"
    section = "Overall view"

    def __init__(self, store, domains: Set[str]):
        super().__init__(store)
        self.domains = set(domains)

    # ------------------------------------------------------------- helpers
    def _chunk(self, ticker: str, prefix: str):
        for c in self.store.chunks.values():
            if c.ticker == ticker and c.source.startswith(prefix):
                return c
        return None

    def _ids(self, *chunks) -> List[str]:
        return [c.chunk_id for c in chunks if c is not None]

    # ----------------------------------------------------------------- run
    def run(self, ticker: str, context: dict) -> List[Claim]:
        if len(self.domains) < 2:
            return []                      # nothing to reconcile across a single domain
        ind = context.get("indicators") or {}
        out: List[Claim] = []
        trend_chunk = self._chunk(ticker, f"derived_{ticker}_trend")
        risk_chunk = self._chunk(ticker, f"derived_{ticker}_risk")
        sent_chunk = self._chunk(ticker, f"derived_{ticker}_sentiment")
        fin_chunk = self._chunk(ticker, "financials_")

        have_tech = "technical" in self.domains and ind and trend_chunk is not None
        have_risk = "risk" in self.domains and ind and risk_chunk is not None
        net: Optional[float] = None
        if "sentiment" in self.domains and sent_chunk is not None:
            v = _find_value(sent_chunk.text, r"Net sentiment score is")
            net = float(v) if v else None
        growth = None
        if "fundamental" in self.domains and fin_chunk is not None:
            g = _find_value(fin_chunk.text, r"Revenue growth over the prior fiscal year is")
            growth = float(g.rstrip("%")) if g else None

        # 1. what the price action says, in one verifiable sentence
        trend = None
        if have_tech:
            trend = _trend(ind)
            r = ind["rsi_14"]
            zone = "overbought" if r >= 70 else "oversold" if r <= 30 else "neutral"
            macd = "above" if ind["macd_line"] > ind["macd_signal"] else "below"
            if trend == "mixed":
                ma = ("above its 20-day moving average but below its 50-day moving average"
                      if ind["last_close"] > ind["sma_20"] else
                      "below its 20-day moving average but above its 50-day moving average")
            else:
                ma = f"{trend} both its 20-day and 50-day moving averages"
            out.append(self.claim(
                f"On price action, the stock closed {ma}, RSI(14) of {r} is in {zone} "
                f"territory, and the MACD line is {macd} its signal line.",
                self._ids(trend_chunk)))

        # 2. news tone vs price trend: agree or diverge
        if net is not None and trend is not None:
            tone = _tone(net)
            if trend == "mixed" or tone == "neutral":
                verdict = "do not point clearly in the same direction"
            elif (tone == "positive") == (trend == "above"):
                verdict = "point in the same direction"
            else:
                verdict = "point in opposite directions"
            where = ("above both moving averages" if trend == "above" else
                     "below both moving averages" if trend == "below" else
                     "between its moving averages")
            out.append(self.claim(
                f"News tone is {tone} (net sentiment score is {net}) while the price closed "
                f"{where}, so the sentiment and trend readings {verdict}.",
                self._ids(sent_chunk, trend_chunk)))

        # 3. the cost of holding it: volatility and the worst fall
        if have_risk:
            out.append(self.claim(
                f"Over the window the annualised volatility was {ind['annual_vol_pct']}% and "
                f"the maximum drawdown was {ind['max_drawdown_pct']}%, the worst peak-to-trough "
                f"fall.", self._ids(risk_chunk)))

        # 4. reported growth vs what the market did with the stock
        if growth is not None and have_risk:
            ret = ind["period_return_pct"]
            rel = ("moved the same way as" if (growth > 0) == (ret > 0)
                   else "moved against")
            out.append(self.claim(
                f"Revenue growth over the prior fiscal year was {growth}% while the total "
                f"price return over the window was {ret}%, so the share price {rel} reported "
                f"growth.", self._ids(fin_chunk, risk_chunk)))
        return out
