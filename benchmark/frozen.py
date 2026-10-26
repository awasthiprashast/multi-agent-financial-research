"""
benchmark/frozen.py — freeze a company's source data, and rebuild a pipeline from it.

The benchmark needs the data a chatbot answer is judged against to stay still: live
prices move every day, so a label written today would be wrong tomorrow. `freeze` saves
exactly what the pipeline ingests (the raw price series and every evidence chunk);
`FrozenPipeline` rebuilds the same store from that file with no network access.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from core import Chunk
from data.corpus import load_corpus
from pipeline import ResearchPipeline

FROZEN_DIR = Path(__file__).resolve().parent / "frozen"


def freeze(ticker: str, out_dir: Path = FROZEN_DIR) -> Path:
    """Fetch live data for `ticker` once and write it to <out_dir>/<ticker>.json."""
    chunks, prices, news_gold, mode = load_corpus(ticker, prefer_live=True)
    if mode != "live":
        raise RuntimeError(f"{ticker}: live data unavailable, refusing to freeze a snapshot")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}.json"
    path.write_text(json.dumps({
        "ticker": ticker, "frozen_on": time.strftime("%Y-%m-%d"), "mode": mode,
        "prices": prices, "news_gold": news_gold,
        "chunks": [asdict(c) for c in chunks],
    }, ensure_ascii=False), encoding="utf-8")
    return path


class FrozenPipeline(ResearchPipeline):
    """A pipeline whose data comes from `freeze` files, never from the network."""

    def __init__(self, frozen_dir: Path = FROZEN_DIR, **kw):
        kw.setdefault("prefer_live", True)           # all frozen companies are live-mode ones
        kw.setdefault("analysts", "template")
        super().__init__(**kw)
        self.frozen_dir = Path(frozen_dir)
        self._frozen: Dict[str, dict] = {}

    @property
    def supported(self):
        return sorted(p.stem for p in self.frozen_dir.glob("*.json"))

    def _build(self, ticker: str) -> dict:
        t0 = time.perf_counter()
        data = json.loads((self.frozen_dir / f"{ticker}.json").read_text(encoding="utf-8"))
        chunks = [Chunk(**c) for c in data["chunks"]]
        return self._assemble(ticker, chunks, data["prices"], data["news_gold"], data["mode"], t0)


def data_sheet(ctx: dict) -> list:
    """The frozen facts a sentence about this company may be checked against, as text lines."""
    ind = ctx["indicators"]
    lines = [f"last close {ind['last_close']}; 20d SMA {ind['sma_20']}; 50d SMA {ind['sma_50']}; "
             f"200d SMA {ind['sma_200']}; RSI(14) {ind['rsi_14']}; MACD {ind['macd_line']} vs signal "
             f"{ind['macd_signal']} (hist {ind['macd_hist']}); Bollinger upper {ind['bb_upper']} lower "
             f"{ind['bb_lower']}; annualised vol {ind['annual_vol_pct']}%; max drawdown "
             f"{ind['max_drawdown_pct']}%; 1-day VaR95 {ind['var_95_pct']}%; period return "
             f"{ind['period_return_pct']}% over {ind['sessions']} sessions"]
    for ch in ctx["store"].chunks.values():
        if ch.doc_type == "analysis" or ch.source.startswith(("fundamentals_", "financials_", "marketdata_")):
            lines.append(f"[{ch.source}] {ch.text}")
    return lines
