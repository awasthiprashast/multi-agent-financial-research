"""
eval/benchmark_real.py — score the auditor on real chatbot text against hand-assigned labels.

    python -m eval.benchmark_real --split dev     # companies used while building the auditor
    python -m eval.benchmark_real --split test    # held-out companies; run once, after freezing
    python -m eval.benchmark_real                 # everything, writes results/real_text_benchmark.json

Everything is offline: the data each answer is judged against is frozen in benchmark/frozen/.

Reading the numbers. Gold labels are C (correct), W (wrong), N (no verdict possible from the
data sheet) and A (ambiguous, excluded). The auditor answers verified / contradicted (it commits)
or unsupported / uncheckable (it abstains). The two errors that matter are
  false accept  - a WRONG sentence the auditor calls verified (dangerous)
  false reject  - a CORRECT sentence the auditor calls contradicted (annoying)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# deterministic offline backends, set before anything imports the pipeline
os.environ["FINRESEARCH_EMBEDDER"] = "tfidf"
os.environ["FINRESEARCH_SCORER"] = "lexicon"
os.environ["FINRESEARCH_LLM"] = "none"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.auditor import CONTRADICTED, UNCHECKABLE, UNSUPPORTED, VERIFIED, audit_text  # noqa: E402
from benchmark.frozen import FrozenPipeline  # noqa: E402
from benchmark.labels import DEV, LABELS, NAMES, TEST  # noqa: E402

PRED = {VERIFIED: "C", CONTRADICTED: "W", UNSUPPORTED: "U", UNCHECKABLE: "N"}

_TYPES = [
    ("rsi", re.compile(r"\brsi\b|relative strength|overbought|oversold", re.I)),
    ("macd", re.compile(r"\bmacd\b", re.I)),
    ("bollinger", re.compile(r"bollinger", re.I)),
    ("moving average", re.compile(r"moving average|\bsma\b|\bema\b|\d-day", re.I)),
    ("risk (volatility, drawdown, VaR)", re.compile(r"volatil|drawdown|\bvar\b|value at risk", re.I)),
    ("valuation (P/E)", re.compile(r"p/e|price-to-earnings|valuation|earnings multiple", re.I)),
    ("profitability and financials",
     re.compile(r"margin|\broe\b|return on equity|profit|revenue|income|\beps\b|debt|market cap", re.I)),
    ("sentiment", re.compile(r"sentiment|news", re.I)),
    ("price level", re.compile(r"trading|price|close|range|₹|inr", re.I)),
]


def kind_of(text: str) -> str:
    for name, pat in _TYPES:
        if pat.search(text):
            return name
    return "other"


def load_answers() -> list:
    answers = json.loads((ROOT / "benchmark" / "raw_answers.json").read_text(encoding="utf-8"))
    for a in answers:
        a.setdefault("setting", "ungrounded")
    return answers


def run(tickers: list) -> list:
    """One row per sentence: id, text, gold code, auditor code, reason."""
    pipe = FrozenPipeline()
    rows = []
    for a in load_answers():
        if a["ticker"] not in tickers:
            continue
        codes = LABELS[(a["ticker"], a["model"], a["setting"])]
        result = audit_text(a["answer"], pipe, ticker=a["ticker"])
        if len(result.sentences) != len(codes):
            raise SystemExit(f"{a['ticker']} {a['model']} {a['setting']}: {len(result.sentences)} "
                             f"sentences but {len(codes)} labels; regenerate benchmark/label_sheet.md")
        for i, (s, g) in enumerate(zip(result.sentences, codes), 1):
            rows.append({"id": f"{a['ticker']}|{a['model']}|{a['setting']}|{i}", "ticker": a["ticker"],
                         "model": a["model"], "setting": a["setting"], "text": s.text, "gold": g,
                         "pred": PRED[s.status], "status": s.status,
                         "reason": s.explanation, "kind": kind_of(s.text)})
    return rows


def _rate(num: int, den: int):
    return round(num / den, 3) if den else None


def metrics(rows: list) -> dict:
    scored = [r for r in rows if r["gold"] != "A"]
    n_amb = len(rows) - len(scored)
    conf = Counter((r["gold"], r["pred"]) for r in scored)
    gold_w = [r for r in scored if r["gold"] == "W"]
    gold_c = [r for r in scored if r["gold"] == "C"]
    gold_n = [r for r in scored if r["gold"] in ("N", "P")]   # no clean verdict possible
    pred_w = [r for r in scored if r["pred"] == "W"]
    pred_c = [r for r in scored if r["pred"] == "C"]
    committed_checkable = [r for r in gold_w + gold_c if r["pred"] in ("C", "W")]
    right = [r for r in committed_checkable if r["pred"] == r["gold"]]
    return {
        "sentences": len(rows), "ambiguous_excluded": n_amb, "scored": len(scored),
        "gold": {"correct": len(gold_c), "wrong": len(gold_w), "no_clean_verdict": len(gold_n)},
        "confusion": {f"gold {NAMES[g]} -> auditor {p}": conf[(g, p)]
                      for g in "CWPN" for p in "CWUN" if conf[(g, p)]},
        "errors_caught": {"recall": _rate(sum(r["pred"] == "W" for r in gold_w), len(gold_w)),
                          "precision": _rate(sum(r["gold"] == "W" for r in pred_w), len(pred_w))},
        "false_accept_rate": _rate(sum(r["pred"] == "C" for r in gold_w), len(gold_w)),
        "false_accepts": [r["id"] for r in gold_w if r["pred"] == "C"],
        "correct_recognised": {"recall": _rate(sum(r["pred"] == "C" for r in gold_c), len(gold_c)),
                               "precision": _rate(sum(r["gold"] == "C" for r in pred_c), len(pred_c))},
        "false_reject_rate": _rate(sum(r["pred"] == "W" for r in gold_c), len(gold_c)),
        "false_rejects": [r["id"] for r in gold_c if r["pred"] == "W"],
        "coverage_of_checkable": _rate(len(committed_checkable), len(gold_w) + len(gold_c)),
        "accuracy_when_committed": _rate(len(right), len(committed_checkable)),
        "overreach_on_no_clean_verdict": _rate(sum(r["pred"] in ("C", "W") for r in gold_n), len(gold_n)),
    }


def breakdown(rows: list, key: str) -> dict:
    out = {}
    for value in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == value]
        m = metrics(sub)
        out[value] = {"scored": m["scored"], "gold": m["gold"],
                      "errors_caught_recall": m["errors_caught"]["recall"],
                      "false_accept_rate": m["false_accept_rate"],
                      "correct_recall": m["correct_recognised"]["recall"],
                      "false_reject_rate": m["false_reject_rate"]}
    return out


def naive_baseline(rows: list) -> dict:
    """What a verifier that trusts any sentence containing a number would score."""
    fake = [dict(r, pred="C" if re.search(r"\d", r["text"]) else "N") for r in rows]
    m = metrics(fake)
    return {"errors_caught_recall": m["errors_caught"]["recall"],
            "false_accept_rate": m["false_accept_rate"],
            "correct_recall": m["correct_recognised"]["recall"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--out", default=None, help="write the full report to this JSON file")
    args = ap.parse_args()
    tickers = {"dev": DEV, "test": TEST, "all": DEV + TEST}[args.split]
    rows = run(tickers)
    report = {"split": args.split, "companies": tickers, **metrics(rows),
              "by_setting": breakdown(rows, "setting"), "by_model": breakdown(rows, "model"),
              "by_kind": breakdown(rows, "kind"), "naive_baseline": naive_baseline(rows)}
    if args.split == "all":
        report["dev"] = metrics([r for r in rows if r["ticker"] in DEV])
        report["test"] = metrics([r for r in rows if r["ticker"] in TEST])
    report["mistakes"] = [{k: r[k] for k in ("id", "gold", "pred", "text", "reason")}
                          for r in rows if r["gold"] in "CW" and r["pred"] in "CW" and r["pred"] != r["gold"]]
    report["missed_by_abstaining"] = [{k: r[k] for k in ("id", "gold", "pred", "text")}
                                      for r in rows if r["gold"] in "CW" and r["pred"] in "UN"]
    text = json.dumps(report, indent=1, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("mistakes", "missed_by_abstaining")}, indent=1, ensure_ascii=False))
    print(f"\n{len(report['mistakes'])} committed mistakes, "
          f"{len(report['missed_by_abstaining'])} checkable sentences left unchecked")


if __name__ == "__main__":
    main()
