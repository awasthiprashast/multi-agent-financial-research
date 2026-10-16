"""
eval/llm_eval.py — does the Verification Agent catch real LLM mistakes?

The perturbation benchmark (eval/perturb.py) injects errors the validator was
designed around. This script instead lets a language model write the research
claims (LLM analysts, agents/llm_analysts.py) and measures what the verifier does
with them — errors nobody injected.

Ground truth for the numeric dimension is independent of the verifier's text
matching: every quantity a claim states is compared with the *structured* source
values (the ground-truth table, the indicators computed from raw prices, the
sentiment counts), not with retrieved text. Qualitative errors (wrong reading of
a true number, unsupported commentary) have no automatic oracle, so every claim
is written to results/llm_claims.csv for manual audit.

Run:  python -m eval.llm_eval            (needs Groq key or a running Ollama)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from typing import Dict, List

from core import Claim, extract_numbers

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")
QUERY = "Complete analysis of {name}"
NAMES = {"TCS.NS": "TCS", "RELIANCE.NS": "Reliance", "INFY.NS": "Infosys"}
TOL = 0.005


def _truth_values(ctx: dict, ticker: str) -> List[float]:
    """Every quantity that is actually true for this ticker, from structured data."""
    from data.corpus import GROUND_TRUTH

    vals: List[float] = []
    vals += [float(v) for v in GROUND_TRUTH.get(ticker, {}).values()]
    vals += [float(v) for v in ctx["indicators"].values() if v == v]
    det = ctx["sentiment_detail"].get(ticker, {})
    vals += [float(v) for v in det.get("counts", {}).values()]
    if "net" in det:
        vals.append(float(det["net"]))
    closes = ctx["prices"]["close"]
    vals += [float(closes[0]), float(closes[-1]), float(max(closes)), float(min(closes)),
             float(len(closes)), float(len(det.get("per_chunk", {})))]
    # window lengths, confidence levels and RSI thresholds: definitions, not data
    vals += [14.0, 20.0, 50.0, 95.0, 1.0, 12.0, 26.0, 9.0, 30.0, 70.0, 100.0]
    return vals


_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SOURCE_ID = re.compile(r"\b[\w.]*(?:news|AR\d+|Q\d+FY\d+|derived|marketdata)[\w.#-]*", re.I)


def _bad_numbers(claim_text: str, truth: List[float]) -> List[str]:
    """Numbers in the claim that match no structured source value.

    A number is accepted if either its face value (70 bps, 240,893 cr) or its
    unit-normalised value matches, ignoring sign: the oracle asks "is this a real
    quantity of this company", not "is it phrased the right way round"."""
    text = _SOURCE_ID.sub(" ", _ISO_DATE.sub(" ", claim_text))
    bad = []
    for fact in extract_numbers(text):
        cands = {abs(fact.value), abs(fact.normalised)}
        ok = any((t == 0 and c == 0) or (t != 0 and abs(abs(t) - c) / abs(t) <= TOL)
                 for t in truth for c in cands)
        if not ok:
            bad.append(fact.raw)
    return bad


def summarize(prefix: str = "") -> None:
    """Aggregate every results/<prefix>llm_eval*.json (one per sampling seed)."""
    import glob

    runs = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, f"{prefix}llm_eval*.json"))):
        if os.path.basename(path) == f"{prefix}llm_eval_summary.json":
            continue
        with open(path, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    keys = ["llm_claims", "claims_flagged_first_pass", "claims_with_a_wrong_number_per_oracle",
            "wrong_numbers_that_reached_the_report", "revisions_attempted",
            "revisions_successful"]
    total = {k: sum(r.get(k, 0) for r in runs) for k in keys}
    total["runs"] = len(runs)
    total["seeds"] = [r.get("seed", 7) for r in runs]
    total["first_pass_flag_rate"] = round(
        total["claims_flagged_first_pass"] / total["llm_claims"], 4) if total["llm_claims"] else None
    with open(os.path.join(RESULTS_DIR, f"{prefix}llm_eval_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(total, fh, indent=2)
    print(json.dumps(total, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true",
                    help="aggregate existing per-seed results and exit")
    ap.add_argument("--groq", action="store_true",
                    help="with --summarize: aggregate the Groq runs instead of Ollama")
    ap.add_argument("--tickers", nargs="*", default=list(NAMES))
    ap.add_argument("--embedder", default="tfidf")
    ap.add_argument("--seed", type=int, default=7, help="sampling seed for the LLM")
    args = ap.parse_args()
    if args.summarize:
        summarize("groq_" if args.groq else "")
        return
    os.environ["FINRESEARCH_LLM_SEED"] = str(args.seed)
    os.environ["FINRESEARCH_EMBEDDER"] = args.embedder
    os.environ["FINRESEARCH_SCORER"] = "lexicon"

    from pipeline import ResearchPipeline

    pipe = ResearchPipeline(prefer_live=False, analysts="llm")
    print("LLM backend:", pipe.llm.name)
    all4 = ["fundamental", "technical", "sentiment", "risk"]

    rows: List[Dict] = []
    stats = {"revisions_attempted": 0, "revisions_successful": 0}
    t0 = time.perf_counter()
    for ticker in args.tickers:
        rep = pipe.run(QUERY.format(name=NAMES[ticker]), ticker=ticker, force_agents=all4)
        ctx = pipe._prepare(ticker)
        truth = _truth_values(ctx, ticker)
        for k in stats:
            stats[k] += rep.loop_stats.get(k, 0)
        final = {c.claim_id: rep.verdicts[c.claim_id] for sec in rep.sections.values() for c in sec}
        surfaced_ids = set(final)
        for claim, v0 in rep.first_pass:
            bad = _bad_numbers(claim.text, truth)
            rows.append({
                "ticker": ticker, "agent": claim.agent, "claim": claim.text,
                "first_pass_status": v0.status, "first_pass_confidence": v0.confidence,
                "verifier_reason": v0.reason,
                "oracle_numeric_ok": not bad, "oracle_bad_numbers": "; ".join(bad),
                "surfaced_unrevised": claim.claim_id in surfaced_ids,
            })
        # claims that surfaced only after revision
        for c in (c for sec in rep.sections.values() for c in sec):
            if c.revision:
                bad = _bad_numbers(c.text, truth)
                rows.append({
                    "ticker": ticker, "agent": c.agent, "claim": c.text,
                    "first_pass_status": "revised-output",
                    "first_pass_confidence": rep.verdicts[c.claim_id].confidence,
                    "verifier_reason": "", "oracle_numeric_ok": not bad,
                    "oracle_bad_numbers": "; ".join(bad), "surfaced_unrevised": True,
                })
        print(f"  {ticker}: {len(rep.first_pass)} claims, {len(rep.withheld)} withheld")

    if not rows:
        raise SystemExit(f"{pipe.llm.name} produced no claims for any ticker. Check the "
                         "backend: an invalid API key or an unreachable server is "
                         "logged above as 'LLM call failed'.")
    first = [r for r in rows if r["first_pass_status"] != "revised-output"]
    revised = [r for r in rows if r["first_pass_status"] == "revised-output"]
    bad = [r for r in first if not r["oracle_numeric_ok"]]
    good = [r for r in first if r["oracle_numeric_ok"]]
    flagged = lambda rs: sum(1 for r in rs if r["first_pass_status"] == "flagged")
    leaked = [r for r in first if r["first_pass_status"] != "flagged"
              and not r["oracle_numeric_ok"]]

    metrics = {
        "llm": pipe.llm.name,
        "tickers": args.tickers,
        "llm_claims": len(first),
        "claims_flagged_first_pass": flagged(first),
        "first_pass_flag_rate": round(flagged(first) / len(first), 4) if first else None,
        "claims_with_a_wrong_number_per_oracle": len(bad),
        "wrong_number_rate": round(len(bad) / len(first), 4) if first else None,
        "verifier_recall_on_wrong_numbers": round(flagged(bad) / len(bad), 4) if bad else None,
        "verifier_flags_on_numerically_ok_claims": flagged(good),
        "numerically_ok_claims": len(good),
        "wrong_numbers_that_reached_the_report": len(leaked),
        "revisions_attempted": stats["revisions_attempted"],
        "revisions_successful": stats["revisions_successful"],
        "revised_outputs_with_wrong_numbers": sum(1 for r in revised if not r["oracle_numeric_ok"]),
        "seconds": round(time.perf_counter() - t0, 1),
        "note": ("numeric oracle uses structured source values; qualitative errors need "
                 "manual review of results/llm_claims.csv"),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "" if args.seed == 7 else f"_seed{args.seed}"
    prefix = "groq_" if pipe.llm.name.startswith("groq") else ""   # keep models apart
    metrics["seed"] = args.seed
    with open(os.path.join(RESULTS_DIR, f"{prefix}llm_eval{tag}.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    with open(os.path.join(RESULTS_DIR, f"{prefix}llm_claims{tag}.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
