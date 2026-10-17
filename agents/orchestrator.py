"""
agents/orchestrator.py — the query-aware Router Agent.

Contribution note: routing is a trained classifier over query features, not an
LLM prompt. A question about sentiment on one stock should not trigger a full
fundamental + technical + risk pipeline. Because the router is a real model, its
accuracy is measurable and the latency and token cost it avoids are measurable
too — which is what turns "we route queries" into a reportable technical effect.

Labels are multi-label: a query may need several agents.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

AGENTS = ["fundamental", "technical", "sentiment", "risk"]

# Multi-label training set. Small, hand-labelled, and deliberately visible in the
# repo so a reviewer can audit exactly what the router learned from.
TRAIN: List[Tuple[str, List[str]]] = [
    ("what is the pe ratio and roe of tcs", ["fundamental"]),
    ("show me the balance sheet strength of reliance", ["fundamental"]),
    ("how are infosys margins trending", ["fundamental"]),
    ("is tcs expensive on earnings", ["fundamental"]),
    ("revenue growth for reliance last year", ["fundamental"]),
    ("what is the debt to equity of infosys", ["fundamental"]),
    ("earnings per share of tcs", ["fundamental"]),

    ("is tcs overbought on rsi", ["technical"]),
    ("what does the macd say about reliance", ["technical"]),
    ("show the moving averages for infosys", ["technical"]),
    ("chart setup for tcs", ["technical"]),
    ("support and resistance levels for reliance", ["technical"]),
    ("has infy broken its 200 day average", ["technical"]),

    ("what is the news sentiment on adani green", ["sentiment"]),
    ("how is the market feeling about reliance", ["sentiment"]),
    ("recent headlines tone for infosys", ["sentiment"]),
    ("is coverage of tcs positive or negative", ["sentiment"]),
    ("what are analysts saying about infy", ["sentiment"]),

    ("how volatile is reliance", ["risk"]),
    ("what is the drawdown risk in tcs", ["risk"]),
    ("beta of infosys against the index", ["risk"]),
    ("how risky is holding reliance for a year", ["risk"]),
    ("downside risk for infy", ["risk"]),

    ("should i look at tcs for a one year hold", ["fundamental", "technical", "sentiment", "risk"]),
    ("give me a full research report on reliance", ["fundamental", "technical", "sentiment", "risk"]),
    ("complete analysis of infosys", ["fundamental", "technical", "sentiment", "risk"]),
    ("everything i should know before buying tcs", ["fundamental", "technical", "sentiment", "risk"]),

    ("compare reliance and tcs on fundamentals and risk", ["fundamental", "risk"]),
    ("valuation and volatility of infosys", ["fundamental", "risk"]),
    ("is tcs cheap and is the chart supportive", ["fundamental", "technical"]),
    ("sentiment and price action on reliance", ["technical", "sentiment"]),
    ("news mood and volatility for infy", ["sentiment", "risk"]),
    ("fundamentals and news flow for tcs", ["fundamental", "sentiment"]),

    # --- second tranche -------------------------------------------------
    ("valuation multiples for reliance", ["fundamental"]),
    ("profitability and return ratios of tcs", ["fundamental"]),
    ("how leveraged is the balance sheet of reliance", ["fundamental"]),
    ("operating margin trend at infosys", ["fundamental"]),
    ("book value and earnings quality of tcs", ["fundamental"]),
    ("dividend payout of infosys", ["fundamental"]),
    ("is reliance trading above its intrinsic value on earnings", ["fundamental"]),

    ("momentum indicators for reliance", ["technical"]),
    ("bollinger bands on tcs", ["technical"]),
    ("is infy trending above its 50 day moving average", ["technical"]),
    ("rsi reading for reliance right now", ["technical"]),
    ("price momentum and crossover signals for tcs", ["technical"]),
    ("where is resistance on the infosys chart", ["technical"]),
    ("technical setup on reliance", ["technical"]),

    ("headline tone around reliance this month", ["sentiment"]),
    ("media coverage mood for tcs", ["sentiment"]),
    ("are the news articles on infy bullish or bearish", ["sentiment"]),
    ("public perception of reliance in the press", ["sentiment"]),
    ("tone of recent commentary on infosys", ["sentiment"]),
    ("news flow sentiment score for tcs", ["sentiment"]),

    ("value at risk for tcs", ["risk"]),
    ("worst peak to trough fall in infosys", ["risk"]),
    ("standard deviation of returns for reliance", ["risk"]),
    ("how large is the downside exposure in tcs", ["risk"]),
    ("volatility profile of infosys", ["risk"]),
    ("tail risk in holding reliance", ["risk"]),

    ("full due diligence on infosys before i invest", ["fundamental", "technical", "sentiment", "risk"]),
    ("deep dive research note on tcs", ["fundamental", "technical", "sentiment", "risk"]),
    ("assess reliance across every dimension", ["fundamental", "technical", "sentiment", "risk"]),
    ("give me the whole picture on infy", ["fundamental", "technical", "sentiment", "risk"]),

    ("earnings quality and drawdown history of infosys", ["fundamental", "risk"]),
    ("balance sheet strength and volatility of tcs", ["fundamental", "risk"]),
    ("chart signals and news tone for infosys", ["technical", "sentiment"]),
    ("momentum and media coverage of tcs", ["technical", "sentiment"]),
    ("news tone and downside risk in reliance", ["sentiment", "risk"]),
    ("sentiment plus value at risk for tcs", ["sentiment", "risk"]),
    ("valuation and chart position of reliance", ["fundamental", "technical"]),
    ("margins and moving averages for infy", ["fundamental", "technical"]),
    ("volatility and chart momentum for reliance", ["technical", "risk"]),
    ("rsi and drawdown for infosys", ["technical", "risk"]),
    ("financials and press coverage of reliance", ["fundamental", "sentiment"]),
    ("roe and analyst commentary on infy", ["fundamental", "sentiment"]),
]

# Company recognition lives in data/universe.py (one list for the router, the
# verifier, the corpus and the API); re-exported here for existing imports.
from data.universe import TICKER_ALIASES, resolve_ticker  # noqa: E402,F401

# --------------------------------------------------------------------------
# Feature design
#
# A bag-of-words model over a few dozen training queries overfits to the exact
# vocabulary it saw. What actually determines routing is *which analytical
# vocabulary a query uses* and *how wide its scope marker is*, so the router is
# given those directly as engineered features. The result is a model with ~13
# parameters per agent instead of thousands, which is why it generalises from a
# small hand-labelled set — and why its cost is a rounding error next to an LLM
# routing call.
# --------------------------------------------------------------------------

LEXICON: Dict[str, List[str]] = {
    "fundamental": [
        "pe", "p/e", "valuation", "valuations", "multiple", "multiples", "roe",
        "return on equity", "margin", "margins", "revenue", "earnings", "eps",
        "balance sheet", "debt", "leverage", "leveraged", "dividend", "payout",
        "profitability", "book value", "financials", "fundamental", "fundamentals",
        "cheap", "expensive", "intrinsic", "growth", "quality", "cash flow",
    ],
    "technical": [
        "rsi", "macd", "moving average", "50 day", "200 day", "chart", "charts",
        "technical", "momentum", "overbought", "oversold", "bollinger", "band",
        "bands", "support", "resistance", "crossover", "trend", "trending",
        "price action", "breakout", "setup", "signals",
    ],
    "sentiment": [
        "sentiment", "news", "headline", "headlines", "media", "coverage",
        "press", "tone", "mood", "bullish", "bearish", "analysts", "commentary",
        "perception", "feeling", "articles", "saying", "news flow",
    ],
    "risk": [
        "risk", "risky", "volatile", "volatility", "drawdown", "beta", "var",
        "value at risk", "downside", "exposure", "tail", "standard deviation",
        "deviation", "worst", "loss", "safe",
    ],
}

BREADTH_MARKERS = [
    "full", "complete", "everything", "whole", "overall", "deep dive",
    "due diligence", "research report", "research note", "all", "comprehensive",
    "should i", "before i invest", "every dimension", "picture", "analysis of",
]


def featurise(query: str) -> List[float]:
    q = " " + query.lower().strip() + " "
    feats: List[float] = []
    for agent in AGENTS:
        terms = LEXICON[agent]
        hits = sum(1 for t in terms if t in q)
        feats.append(float(hits))                       # raw count
        feats.append(1.0 if hits else 0.0)              # presence
    breadth = sum(1 for m in BREADTH_MARKERS if m in q)
    feats.append(float(breadth))
    feats.append(1.0 if breadth else 0.0)
    feats.append(len(q.split()) / 10.0)                 # query length
    feats.append(1.0 if (" and " in q or "compare" in q or " plus " in q) else 0.0)
    total_hits = sum(1 for a in AGENTS for t in LEXICON[a] if t in q)
    feats.append(float(total_hits))
    return feats


class Router:
    """One-vs-rest logistic regression over engineered query features."""

    def __init__(self, C: float = 2.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer

        self._mlb = MultiLabelBinarizer(classes=AGENTS)
        self._clf = OneVsRestClassifier(
            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        self._fitted = False

    def fit(self, data=None) -> "Router":
        data = data or TRAIN
        xs = [featurise(q) for q, _ in data]
        ys = self._mlb.fit_transform([labels for _, labels in data])
        self._clf.fit(xs, ys)
        self._fitted = True
        return self

    @property
    def n_parameters(self) -> int:
        return int(sum(e.coef_.size + e.intercept_.size
                       for e in self._clf.estimators_))

    def route(self, query: str, threshold: float = 0.50) -> List[str]:
        if not self._fitted:
            self.fit()
        proba = self._clf.predict_proba([featurise(query)])[0]
        chosen = [a for a, p in zip(AGENTS, proba) if p >= threshold]
        if not chosen:                       # never return an empty pipeline
            chosen = [AGENTS[int(proba.argmax())]]
        return chosen

    def scores(self, query: str) -> Dict[str, float]:
        if not self._fitted:
            self.fit()
        proba = self._clf.predict_proba([featurise(query)])[0]
        return {a: float(p) for a, p in zip(AGENTS, proba)}
