"""
eval/run_all.py — the full Review-2 evaluation.

Writes results/metrics.json and results/*.csv. Everything reported in the
presentation and the report comes out of this one script, so any number can be
reproduced with:  python -m eval.run_all --offline
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from typing import Dict, List

from agents.orchestrator import TRAIN, Router
from agents.validator import VerificationAgent
from core import Claim
from data.corpus import GROUND_TRUTH, all_tickers
from eval.perturb import build_benchmark
from pipeline import ResearchPipeline

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

QUERIES = [
    "Should I look at TCS for a 1-year hold?",
    "Give me a full research report on Reliance",
    "Complete analysis of Infosys",
    "Is TCS cheap and is the chart supportive?",
    "How volatile is Reliance?",
    "What is the news sentiment on Infosys?",
    "Fundamentals and news flow for TCS",
    "Compare Reliance and TCS on fundamentals and risk",
]


def _owner(ticker_store: Dict, claim: Claim) -> str:
    """The ticker a claim was generated from, read off its cited chunks."""
    for t, st in ticker_store.items():
        if any(st.get(cid) is not None for cid in claim.cited_chunk_ids):
            return t
    return ""


# ---------------------------------------------------------------- 1. grounding
def eval_grounding(pipe: ResearchPipeline) -> Dict:
    per_report, all_claims = [], []
    for q in QUERIES:
        rep = pipe.run(q)
        surfaced = rep.surfaced_count
        per_report.append({
            "query": q,
            "ticker": rep.ticker,
            "agents": len(rep.agents_invoked),
            "claims_generated": len(rep.verdicts),
            "claims_surfaced": surfaced,
            "grounding_rate": round(rep.grounding_rate, 4),
            "mean_confidence": round(statistics.mean(
                [v.confidence for v in rep.verdicts.values()]), 4),
            "total_s": rep.timings["total_s"],
            "verify_s": rep.timings["verify_s"],
        })
        for claims in rep.sections.values():
            all_claims.extend(claims)

    total_gen = sum(r["claims_generated"] for r in per_report)
    total_srf = sum(r["claims_surfaced"] for r in per_report)
    return {
        "reports": len(per_report),
        "claims_generated": total_gen,
        "claims_surfaced": total_srf,
        "grounding_rate": round(total_srf / total_gen, 4) if total_gen else 0.0,
        "mean_confidence": round(statistics.mean(
            [r["mean_confidence"] for r in per_report]), 4),
        "per_report": per_report,
        "_claims": all_claims,
    }


# ---------------------------------------------------- 2. hallucination catching
def eval_hallucination(pipe: ResearchPipeline, claims: List[Claim]) -> Dict:
    """Positive class = 'unsupported'. The validator sees no labels."""
    ticker_store = {}
    for t in all_tickers():
        ticker_store[t] = pipe._prepare(t)["store"]

    # attribute each claim to the store it came from via its cited chunk ids
    def store_for(c: Claim):
        for t, st in ticker_store.items():
            if any(st.get(cid) is not None for cid in c.cited_chunk_ids):
                return st
        return ticker_store[all_tickers()[0]]

    bench = build_benchmark(claims, seed=7,
                            owner_of=lambda c: _owner(ticker_store, c))
    tp = fp = tn = fn = 0
    by_kind: Dict[str, Counter] = defaultdict(Counter)
    rows = []

    t0 = time.perf_counter()
    for item in bench:
        c: Claim = item["claim"]
        validator = VerificationAgent(store_for(c))
        v = validator.verify(c)
        predicted = "unsupported" if v.status == "flagged" else "supported"
        gold = item["gold"]
        by_kind[item["kind"]][("hit" if predicted == gold else "miss")] += 1
        if gold == "unsupported" and predicted == "unsupported":
            tp += 1
        elif gold == "supported" and predicted == "unsupported":
            fp += 1
        elif gold == "supported" and predicted == "supported":
            tn += 1
        else:
            fn += 1
        rows.append({
            "kind": item["kind"], "gold": gold, "predicted": predicted,
            "confidence": v.confidence, "numeric_support": v.numeric_support,
            "semantic_support": v.semantic_support,
            "text": c.text[:120],
        })
    elapsed = time.perf_counter() - t0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "benchmark_size": len(bench),
        "true_positive": tp, "false_positive": fp,
        "true_negative": tn, "false_negative": fn,
        "catch_rate_recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp + tn) / len(bench), 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "ms_per_claim": round(1000 * elapsed / len(bench), 2),
        "by_perturbation": {k: {"caught": v["hit"], "missed": v["miss"],
                                "rate": round(v["hit"] / (v["hit"] + v["miss"]), 4)}
                            for k, v in by_kind.items()},
        "_rows": rows,
    }


# ---------------------------------------------------------------- 2b. ablation
def eval_ablation(pipe: ResearchPipeline, claims: List[Claim]) -> Dict:
    """Turn each component of the verification mechanism off, one at a time.

    A verification mechanism that scores the same with a component removed does
    not need that component. This table is what justifies the design.
    """
    ticker_store = {t: pipe._prepare(t)["store"] for t in all_tickers()}

    def store_for(c: Claim):
        for st in ticker_store.values():
            if any(st.get(cid) is not None for cid in c.cited_chunk_ids):
                return st
        return next(iter(ticker_store.values()))

    bench = build_benchmark(claims, seed=7,
                            owner_of=lambda c: _owner(ticker_store, c))

    configs = {
        "full mechanism": dict(numeric_veto=True, entity_check=True),
        "no numeric veto (additive scoring)": dict(numeric_veto=False, entity_check=True),
        "no entity binding": dict(numeric_veto=True, entity_check=False),
        "semantic similarity only": dict(numeric_veto=False, entity_check=False,
                                         logic_check=False,
                                         w_numeric=0.0, w_semantic=1.0, w_citation=0.0),
        "citation trust only (no re-retrieval check)":
            dict(numeric_veto=False, entity_check=False, logic_check=False,
                 w_numeric=0.0, w_semantic=0.0, w_citation=1.0),
    }

    out = {}
    for label, kwargs in configs.items():
        tp = fp = tn = fn = 0
        for item in bench:
            c: Claim = item["claim"]
            v = VerificationAgent(store_for(c), **kwargs).verify(c)
            predicted = "unsupported" if v.status == "flagged" else "supported"
            gold = item["gold"]
            if gold == "unsupported" and predicted == "unsupported":
                tp += 1
            elif gold == "supported" and predicted == "unsupported":
                fp += 1
            elif gold == "supported" and predicted == "supported":
                tn += 1
            else:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[label] = {
            "catch_rate": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
            "accuracy": round((tp + tn) / len(bench), 4),
            "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        }
    return out


# -------------------------------------------------------- 2c. threshold sweep
def eval_threshold_sweep(pipe: ResearchPipeline, claims: List[Claim]) -> List[Dict]:
    """How the confidence floor θ trades catch rate against claims surfaced."""
    ticker_store = {t: pipe._prepare(t)["store"] for t in all_tickers()}

    def store_for(c: Claim):
        for st in ticker_store.values():
            if any(st.get(cid) is not None for cid in c.cited_chunk_ids):
                return st
        return next(iter(ticker_store.values()))

    bench = build_benchmark(claims, seed=7,
                            owner_of=lambda c: _owner(ticker_store, c))
    # score once, threshold many times
    scored = []
    for item in bench:
        c = item["claim"]
        v = VerificationAgent(store_for(c), threshold=0.0).verify(c)
        scored.append((v.confidence, item["gold"]))

    rows = []
    for theta in [round(x * 0.05, 2) for x in range(4, 19)]:
        tp = fp = tn = fn = 0
        for conf, gold in scored:
            predicted = "unsupported" if conf < theta else "supported"
            if gold == "unsupported" and predicted == "unsupported":
                tp += 1
            elif gold == "supported" and predicted == "unsupported":
                fp += 1
            elif gold == "supported" and predicted == "supported":
                tn += 1
            else:
                fn += 1
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({
            "theta": theta,
            "catch_rate": round(rec, 4),
            "precision": round(prec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
            "genuine_claims_retained": round(tn / (tn + fp), 4) if (tn + fp) else 0.0,
        })
    return rows


# ------------------------------------------------------------------- 3. router
def eval_router() -> Dict:
    """Leave-one-out evaluation of the router, plus the work it avoids."""
    correct_exact = 0
    jaccards, hits, total_labels = [], 0, 0
    agents_selected, agents_if_always_all = 0, 0

    for i in range(len(TRAIN)):
        train = TRAIN[:i] + TRAIN[i + 1:]
        q, gold = TRAIN[i]
        r = Router().fit(train)
        pred = set(r.route(q))
        gold_set = set(gold)
        if pred == gold_set:
            correct_exact += 1
        inter = len(pred & gold_set)
        union = len(pred | gold_set)
        jaccards.append(inter / union if union else 1.0)
        hits += inter
        total_labels += len(gold_set)
        agents_selected += len(pred)
        agents_if_always_all += 4

    return {
        "queries": len(TRAIN),
        "exact_match_accuracy": round(correct_exact / len(TRAIN), 4),
        "label_recall": round(hits / total_labels, 4),
        "mean_jaccard": round(statistics.mean(jaccards), 4),
        "agents_invoked_total": agents_selected,
        "agents_if_no_routing": agents_if_always_all,
        "agent_invocation_reduction_pct": round(
            100 * (1 - agents_selected / agents_if_always_all), 2),
    }


def eval_router_heldout() -> Dict:
    """Router on queries written after it was frozen (eval/router_testset.py)."""
    from eval.router_testset import TEST

    r = Router().fit()
    exact = hits = total = selected = 0
    jaccards = []
    for q, gold in TEST:
        pred, g = set(r.route(q)), set(gold)
        exact += int(pred == g)
        hits += len(pred & g)
        total += len(g)
        selected += len(pred)
        jaccards.append(len(pred & g) / len(pred | g))
    return {
        "queries": len(TEST),
        "exact_match_accuracy": round(exact / len(TEST), 4),
        "label_recall": round(hits / total, 4),
        "mean_jaccard": round(statistics.mean(jaccards), 4),
        "agent_invocation_reduction_pct": round(100 * (1 - selected / (4 * len(TEST))), 2),
        "note": "held-out: never used to tune features, lexicon or training data",
    }


# ---------------------------------------------------------------- 4. sentiment
def eval_sentiment(pipe: ResearchPipeline) -> Dict:
    from eval.sentiment_testset import texts_and_labels
    from sentiment_model import get_scorer

    scorer = get_scorer()

    # (a) held-out hand-labelled set — the number that means something
    texts, gold = texts_and_labels()
    preds = scorer.labels(texts)
    correct = sum(int(p == g) for p, g in zip(preds, gold))
    confusion: Dict[str, Counter] = defaultdict(Counter)
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for p, g in zip(preds, gold):
        confusion[g][p] += 1
        if p == g:
            per_class[g]["tp"] += 1
        else:
            per_class[p]["fp"] += 1
            per_class[g]["fn"] += 1
    macro_f1 = []
    for cls in ("positive", "negative", "neutral"):
        d = per_class[cls]
        prec = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) else 0.0
        rec = d["tp"] / (d["tp"] + d["fn"]) if (d["tp"] + d["fn"]) else 0.0
        macro_f1.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)

    # (b) pipeline sanity check on the snapshot corpus — NOT a model result
    snap_correct = snap_total = 0
    for t in all_tickers():
        ctx = pipe._prepare(t)
        store = ctx["store"]
        for cid, g in (ctx["news_gold"] or {}).items():
            ch = store.get(cid)
            if ch is None:
                continue
            snap_total += 1
            snap_correct += int(scorer.label(ch.text) == g)

    return {
        "model": scorer.name,
        "heldout_items": len(texts),
        "heldout_accuracy": round(correct / len(texts), 4),
        "heldout_macro_f1": round(sum(macro_f1) / 3, 4),
        "confusion": {g: dict(c) for g, c in confusion.items()},
        "snapshot_sanity_accuracy": round(snap_correct / snap_total, 4) if snap_total else None,
        "snapshot_sanity_items": snap_total,
        "note": ("held-out accuracy is the reportable figure; the snapshot check only "
                 "confirms the scoring path is wired correctly and is not a model "
                 "benchmark. FinBERT replaces the fallback scorer wherever the model "
                 "weights are available."),
    }


# ------------------------------------------------------------------ 5. latency
def eval_cost_latency(pipe: ResearchPipeline) -> Dict:
    routed, full = [], []
    for q in QUERIES:
        rep = pipe.run(q)
        routed.append(rep.timings)
        rep_full = pipe.run(q, force_agents=["fundamental", "technical",
                                             "sentiment", "risk"])
        full.append(rep_full.timings)

    def mean(rows, key):
        return round(statistics.mean(r[key] for r in rows), 4)

    return {
        "routed_mean_total_s": mean(routed, "total_s"),
        "unrouted_mean_total_s": mean(full, "total_s"),
        "routed_mean_verify_s": mean(routed, "verify_s"),
        "routed_mean_agents_s": mean(routed, "agents_s"),
        "latency_reduction_pct": round(
            100 * (1 - mean(routed, "total_s") / mean(full, "total_s")), 2),
        "external_api_cost_usd_per_report": 0.0,
        "note": ("all inference is local or free-tier; monetary cost per report is "
                 "zero by construction"),
    }


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", default=True)
    ap.add_argument("--live", dest="offline", action="store_false")
    ap.add_argument("--embedder", choices=["tfidf", "neural", "auto"], default="tfidf",
                    help="tfidf (default) is deterministic and reproducible")
    ap.add_argument("--scorer", choices=["lexicon", "finbert", "auto"], default="lexicon")
    ap.add_argument("--out", default="metrics.json",
                    help="file name inside results/ (e.g. metrics_neural.json)")
    args = ap.parse_args()

    # Backends are chosen explicitly so a published number always says which
    # embedder and scorer produced it, regardless of what is installed.
    os.environ["FINRESEARCH_EMBEDDER"] = args.embedder
    os.environ["FINRESEARCH_SCORER"] = args.scorer

    pipe = ResearchPipeline(prefer_live=not args.offline)

    print("[1/5] grounding ...")
    grounding = eval_grounding(pipe)
    claims = grounding.pop("_claims")

    print("[2/5] hallucination catch rate ...")
    halluc = eval_hallucination(pipe, claims)
    rows = halluc.pop("_rows")

    print("[3/6] ablation + threshold sweep ...")
    ablation = eval_ablation(pipe, claims)
    sweep = eval_threshold_sweep(pipe, claims)

    print("[4/6] router ...")
    router = eval_router()
    router_heldout = eval_router_heldout()

    print("[5/6] sentiment ...")
    sentiment = eval_sentiment(pipe)

    print("[6/6] cost & latency ...")
    latency = eval_cost_latency(pipe)

    metrics = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_mode": pipe._prepare(all_tickers()[0])["mode"],
        "embedder": pipe._prepare(all_tickers()[0])["store"].embedder.name,
        "sentiment_scorer": sentiment["model"],
        "retrieval_backend": pipe._prepare(all_tickers()[0])["store"].backend,
        "chunks_indexed": sum(pipe._prepare(t)["chunks"] for t in all_tickers()),
        "grounding": grounding,
        "hallucination": halluc,
        "ablation": ablation,
        "threshold_sweep": sweep,
        "router": router,
        "router_heldout": router_heldout,
        "sentiment": sentiment,
        "cost_latency": latency,
    }

    with open(os.path.join(RESULTS_DIR, args.out), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    import csv
    with open(os.path.join(RESULTS_DIR, "verification_detail.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(json.dumps({k: v for k, v in metrics.items() if k != "grounding"}, indent=2)[:3000])
    print("\ngrounding:", {k: v for k, v in grounding.items() if k != "per_report"})
    print(f"\nwrote results/{args.out} and results/verification_detail.csv")


if __name__ == "__main__":
    main()
