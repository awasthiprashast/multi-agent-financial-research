"""
eval/router_testset.py — held-out queries for the router.

Written after the router and its training set were frozen, and never used to tune
the lexicon, the features or the training data. Phrasing deliberately differs from
the training queries: other company names, informal wording, different word order.
Labels are the agents a human analyst would need to answer the question.
"""

from typing import List, Tuple

TEST: List[Tuple[str, List[str]]] = [
    ("is infosys stock overvalued right now", ["fundamental"]),
    ("tell me about reliance's profit margins", ["fundamental"]),
    ("what are the earnings like at tcs", ["fundamental"]),
    ("how much debt does reliance carry", ["fundamental"]),
    ("tcs return on equity vs infosys", ["fundamental"]),
    ("is the rsi on infy showing overbought", ["technical"]),
    ("where does tcs sit relative to its 50 day line", ["technical"]),
    ("any macd crossover on reliance lately", ["technical"]),
    ("bollinger band squeeze in infosys", ["technical"]),
    ("what are people saying about tcs in the news", ["sentiment"]),
    ("is the press bullish on reliance", ["sentiment"]),
    ("headline mood for infosys this week", ["sentiment"]),
    ("how bad could a drawdown in tcs get", ["risk"]),
    ("is reliance a volatile stock", ["risk"]),
    ("what is the worst one day loss for infy", ["risk"]),
    ("how risky is infosys", ["risk"]),
    ("give me everything on tcs", ["fundamental", "technical", "sentiment", "risk"]),
    ("i am thinking of buying reliance, what should i know", ["fundamental", "technical", "sentiment", "risk"]),
    ("full report on infy please", ["fundamental", "technical", "sentiment", "risk"]),
    ("is tcs a good long term investment", ["fundamental", "technical", "sentiment", "risk"]),
    ("valuation and recent headlines for reliance", ["fundamental", "sentiment"]),
    ("tcs margins and how volatile the shares are", ["fundamental", "risk"]),
    ("infosys chart and drawdown history", ["technical", "risk"]),
    ("reliance rsi and news tone", ["technical", "sentiment"]),
    ("is tcs cheap and what does the market think", ["fundamental", "sentiment"]),
    ("momentum and downside risk for infosys", ["technical", "risk"]),
    ("eps growth and analyst commentary for reliance", ["fundamental", "sentiment"]),
    ("tcs support level", ["technical"]),
    ("infosys dividend", ["fundamental"]),
    ("reliance var", ["risk"]),
]
