"""
Gold labels for the real-text benchmark, one code per sentence in order:

    C  correct       every fact in the sentence that the frozen data sheet can settle agrees
                     with it ("about/around/near" allows 2 % either way; otherwise the figure
                     must match to the precision stated; units and scale count: a market cap
                     of "₹1.63 trillion" is wrong when the sheet says ₹16.3 trillion)
    W  wrong         at least one such fact contradicts the sheet, whatever else the sentence says
    P  partly        every fact the sheet can settle is correct, but the sentence also states
                     figures it cannot settle (forward P/E, P/B, ...): no clean verdict
    N  no verdict    nothing in the sentence can be settled by the sheet: forecasts, advice,
                     peer or historical averages, other fiscal years or quarters, ratios the sheet
                     does not hold (forward P/E, P/B, NIM, dividend yield ...), events
    A  ambiguous     a reasonable reader could label it either way (period unclear, a borderline
                     "around"); excluded from every score and counted in the report

Labelling protocol: each sentence is read against its company's frozen data sheet
(benchmark/label_sheet.md) without looking at the auditor's verdict. The auditor's verdicts are
produced afterwards by eval/benchmark_real.py.

Development / test split: labels were written for all companies up front. The auditor was
tuned only on the DEV companies; the TEST companies were scored once, after the code was frozen.
"""

from __future__ import annotations

DEV = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "ITC.NS"]
TEST = ["LT.NS", "WIPRO.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "ADANIGREEN.NS"]

GPT = "openai/gpt-oss-120b"
QWEN = "qwen/qwen3.8-27b"

# (ticker, model, setting) -> codes
LABELS = {
    # ---- chatbot answers written from memory (no data supplied) ----
    ("TCS.NS", GPT, "ungrounded"): "WWAWNN",
    ("TCS.NS", QWEN, "ungrounded"): "WWNWNNN",
    ("RELIANCE.NS", GPT, "ungrounded"): "WWNNNN",
    ("RELIANCE.NS", QWEN, "ungrounded"): "WWNNNNNN",
    ("INFY.NS", GPT, "ungrounded"): "WWNANN",
    ("INFY.NS", QWEN, "ungrounded"): "WWNWNNNN",
    ("HDFCBANK.NS", GPT, "ungrounded"): "WWWNNN",
    ("HDFCBANK.NS", QWEN, "ungrounded"): "WWNWNNNN",
    ("ICICIBANK.NS", GPT, "ungrounded"): "WNANNN",
    ("ICICIBANK.NS", QWEN, "ungrounded"): "WWWNNNNN",
    ("SBIN.NS", GPT, "ungrounded"): "WWWWNN",
    ("SBIN.NS", QWEN, "ungrounded"): "WWPWNNNN",
    ("ITC.NS", GPT, "ungrounded"): "WWNANN",
    ("ITC.NS", QWEN, "ungrounded"): "WWWWNNNN",
    ("LT.NS", GPT, "ungrounded"): "WWANNNN",
    ("LT.NS", QWEN, "ungrounded"): "CWNNNNNN",
    ("WIPRO.NS", GPT, "ungrounded"): "WWNNNW",
    ("WIPRO.NS", QWEN, "ungrounded"): "WWNWWNNN",
    ("BHARTIARTL.NS", GPT, "ungrounded"): "WWNNNN",
    ("BHARTIARTL.NS", QWEN, "ungrounded"): "WWWWNNNW",
    ("HINDUNILVR.NS", GPT, "ungrounded"): "WWNNNW",
    ("HINDUNILVR.NS", QWEN, "ungrounded"): "WAWWNWNW",
    ("ADANIGREEN.NS", GPT, "ungrounded"): "WWNNNN",
    ("ADANIGREEN.NS", QWEN, "ungrounded"): "WWCNCNNN",
    # ---- chatbot answers written from the data sheet (grounded) ----
    ("TCS.NS", GPT, "grounded"): "CCCCCN",
    ("TCS.NS", QWEN, "grounded"): "WCCCCCCN",
    ("RELIANCE.NS", GPT, "grounded"): "CCWCCCN",
    ("RELIANCE.NS", QWEN, "grounded"): "CCCCCCC",
    ("INFY.NS", GPT, "grounded"): "CCWWCA",
    ("INFY.NS", QWEN, "grounded"): "CCCCCCCN",
    ("HDFCBANK.NS", GPT, "grounded"): "CWWCCN",
    ("HDFCBANK.NS", QWEN, "grounded"): "CCCCCCCC",
    ("ICICIBANK.NS", GPT, "grounded"): "CCCCWCN",
    ("ICICIBANK.NS", QWEN, "grounded"): "CCCCCCCN",
    ("SBIN.NS", GPT, "grounded"): "CWWCCCN",
    ("SBIN.NS", QWEN, "grounded"): "CCCCCCN",
    ("ITC.NS", GPT, "grounded"): "CCCCWCN",
    ("ITC.NS", QWEN, "grounded"): "CCCCCCCN",
    ("LT.NS", GPT, "grounded"): "CCCCCCN",
    ("LT.NS", QWEN, "grounded"): "CCCCCCCN",
    ("WIPRO.NS", GPT, "grounded"): "WCCCWN",
    ("WIPRO.NS", QWEN, "grounded"): "CCCCCCN",
    ("BHARTIARTL.NS", GPT, "grounded"): "CCCCCC",
    ("BHARTIARTL.NS", QWEN, "grounded"): "CCCCCCCN",
    ("HINDUNILVR.NS", GPT, "grounded"): "CCCCCA",
    ("HINDUNILVR.NS", QWEN, "grounded"): "CCCCCCCN",
    ("ADANIGREEN.NS", GPT, "grounded"): "WWCCCCN",
    ("ADANIGREEN.NS", QWEN, "grounded"): "CCCCCCCN",
}

NAMES = {"C": "CORRECT", "W": "WRONG", "N": "NO_VERDICT", "P": "PARTLY_CHECKABLE", "A": "AMBIGUOUS"}
