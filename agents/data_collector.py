"""
agents/data_collector.py — Data Collection Agent.

Pulls prices, filings and news, normalises them into a common schema, and — the
part that matters for grounding — writes *derived* quantities back into the
evidence store as first-class evidence chunks.

Why derived evidence is a chunk: a technical claim like "RSI(14) is 61.2" cannot
be grounded in a raw price table, because the string "61.2" appears nowhere in
it. Rather than exempting computed values from verification, the system records
each computed quantity together with the input window it was computed over. The
validator can then check it exactly like any other claim, and the audit trail
runs all the way back to the raw series.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from core import Chunk


# ------------------------------------------------------------------ indicators
def sma(values: List[float], window: int) -> float:
    if len(values) < window:
        return float("nan")
    return sum(values[-window:]) / window


def rsi(values: List[float], period: int = 14) -> float:
    if len(values) < period + 1:
        return float("nan")
    gains, losses = [], []
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        (gains if diff >= 0 else losses).append(abs(diff))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _ema(values: List[float], span: int) -> List[float]:
    k = 2.0 / (span + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def macd(values: List[float]) -> Tuple[float, float, float]:
    if len(values) < 35:
        return float("nan"), float("nan"), float("nan")
    fast, slow = _ema(values, 12), _ema(values, 26)
    line = [f - s for f, s in zip(fast, slow)]
    signal = _ema(line, 9)
    return line[-1], signal[-1], line[-1] - signal[-1]


def bollinger(values: List[float], window: int = 20, k: float = 2.0) -> Tuple[float, float]:
    if len(values) < window:
        return float("nan"), float("nan")
    win = values[-window:]
    mean = sum(win) / window
    sd = math.sqrt(sum((v - mean) ** 2 for v in win) / window)
    return mean + k * sd, mean - k * sd


def daily_returns(closes: List[float]) -> List[float]:
    return [(closes[i] / closes[i - 1]) - 1.0 for i in range(1, len(closes))]


def annualised_vol(closes: List[float]) -> float:
    rets = daily_returns(closes)
    if len(rets) < 2:
        return float("nan")
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252) * 100.0


def max_drawdown(closes: List[float]) -> float:
    peak, worst = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        worst = min(worst, (c / peak) - 1.0)
    return worst * 100.0


def var_historical(closes: List[float], alpha: float = 0.05) -> float:
    rets = sorted(daily_returns(closes))
    if not rets:
        return float("nan")
    idx = max(0, int(alpha * len(rets)) - 1)
    return rets[idx] * 100.0


# ------------------------------------------------------------------- collector
class DataCollectionAgent:
    name = "data_collector"

    def compute(self, ticker: str, prices: Dict[str, list]) -> Dict[str, float]:
        closes = [float(c) for c in prices["close"]]
        upper, lower = bollinger(closes)
        macd_line, macd_signal, macd_hist = macd(closes)
        return {
            "last_close": round(closes[-1], 2),
            "sma_20": round(sma(closes, 20), 2),
            "sma_50": round(sma(closes, 50), 2),
            "sma_200": round(sma(closes, 200), 2) if len(closes) >= 200 else float("nan"),
            "rsi_14": round(rsi(closes, 14), 1),
            "macd_line": round(macd_line, 2),
            "macd_signal": round(macd_signal, 2),
            "macd_hist": round(macd_hist, 2),
            "bb_upper": round(upper, 2),
            "bb_lower": round(lower, 2),
            "annual_vol_pct": round(annualised_vol(closes), 1),
            "max_drawdown_pct": round(max_drawdown(closes), 1),
            "var_95_pct": round(var_historical(closes), 2),
            "period_return_pct": round((closes[-1] / closes[0] - 1) * 100, 1),
            "sessions": len(closes),
        }

    def derived_evidence(self, ticker: str, ind: Dict[str, float],
                         prices: Dict[str, list]) -> List[Chunk]:
        """Record computed quantities as auditable evidence chunks."""
        window = f"{prices['date'][0]} to {prices['date'][-1]} ({ind['sessions']} sessions)"
        docs = [
            (f"derived_{ticker}_trend",
             f"Computed trend indicators over {window}. Last close ₹{ind['last_close']}. "
             f"20-day simple moving average ₹{ind['sma_20']}. 50-day simple moving "
             f"average ₹{ind['sma_50']}. "
             + (f"200-day simple moving average ₹{ind['sma_200']}. "
                if ind["sma_200"] == ind["sma_200"] else "")
             + f"RSI(14) is {ind['rsi_14']}. MACD line "
             f"{ind['macd_line']} against signal {ind['macd_signal']}, histogram "
             f"{ind['macd_hist']}. Bollinger band upper ₹{ind['bb_upper']}, lower "
             f"₹{ind['bb_lower']}."),
            (f"derived_{ticker}_risk",
             f"Computed risk statistics over {window}. Annualised volatility of daily "
             f"log returns is {ind['annual_vol_pct']}%. Maximum drawdown within the "
             f"window is {ind['max_drawdown_pct']}%. Historical one-day Value at Risk "
             f"at the 95% level is {ind['var_95_pct']}%. Total return over the window "
             f"is {ind['period_return_pct']}%."),
        ]
        out = []
        for src, text in docs:
            out.append(Chunk(
                chunk_id=Chunk.make_id(src, text), text=text, source=src,
                doc_type="market_data", ticker=ticker,
                published=prices["date"][-1],
                meta={"provenance": "computed", "window": window},
            ))
        return out
