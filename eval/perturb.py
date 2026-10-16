"""
eval/perturb.py — the hallucination benchmark generator.

You cannot measure a verification mechanism on true claims alone; a detector that
approves everything scores 100% on them. So the harness constructs a labelled set
in which each claim is either GENUINE (produced by an agent from real evidence,
gold label = supported) or PERTURBED (the same claim with a defect injected, gold
label = unsupported). The validator never sees the labels.

Four perturbation families, chosen because they are the four ways an LLM actually
goes wrong on financial text:

  numeric_drift      a quantity is scaled by a small factor — the dangerous case,
                     because the sentence still reads perfectly
  numeric_swap       a quantity is replaced by a plausible but different value
  entity_swap        the claim is transplanted onto a different company
  fabricated_detail  a confident, specific, wholly unsupported assertion is added

Reported metrics are standard binary-detection metrics with "unsupported" as the
positive class.
"""

from __future__ import annotations

import random
import re
from typing import Dict, List, Tuple

from core import Claim, NUM_RE

PERTURBATIONS = ["numeric_drift", "numeric_swap", "entity_swap", "fabricated_detail"]

FABRICATIONS = [
    " The company also disclosed a one-time impairment of ₹4,180 cr relating to its "
    "European subsidiary.",
    " Management guided to an operating margin of 29.4% for the coming financial year.",
    " The board approved a share buyback of ₹18,000 cr during the same period.",
    " Attrition rose to 27.9% on a trailing twelve-month basis.",
]

ENTITIES = ["Tata Consultancy Services", "Reliance Industries", "Infosys",
            "HDFC Bank", "Adani Green Energy"]


def _first_number_span(text: str):
    for m in NUM_RE.finditer(text):
        raw = m.group("num")
        if raw is None:
            continue
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        if not m.group("unit") and 1900 <= val <= 2100 and val.is_integer():
            continue
        return m, val
    return None, None


TICKER_NAME = {"TCS.NS": "Tata Consultancy Services", "RELIANCE.NS": "Reliance Industries",
               "INFY.NS": "Infosys", "HDFCBANK.NS": "HDFC Bank",
               "ADANIGREEN.NS": "Adani Green Energy"}


def perturb(claim: Claim, kind: str, rng: random.Random,
            owner: str = "") -> Tuple[Claim, str]:
    """Return (perturbed claim, kind actually applied).

    `owner` is the ticker whose evidence the claim was generated from. An entity
    swap must never pick the owner itself: attributing a claim to the company it
    is genuinely about produces a *true* sentence, not a hallucination.
    """
    text = claim.text
    owner_name = TICKER_NAME.get(owner, "")
    pool = [x for x in ENTITIES if x != owner_name]

    if kind in ("numeric_drift", "numeric_swap"):
        m, val = _first_number_span(text)
        if m is None:
            kind = "fabricated_detail"
        else:
            if kind == "numeric_drift":
                factor = rng.choice([1.08, 1.15, 0.88, 0.82])
                new_val = val * factor
            else:
                new_val = val * rng.choice([1.9, 0.45, 2.6])
            fmt = f"{new_val:,.2f}" if abs(new_val) < 1000 else f"{new_val:,.0f}"
            fmt = fmt.rstrip("0").rstrip(".") if "." in fmt else fmt
            start, end = m.span("num")
            text = text[:start] + fmt + text[end:]

    if kind == "entity_swap":
        replaced = False
        for e in ENTITIES:
            if e in text:
                alt = rng.choice([x for x in pool if x != e])
                text = text.replace(e, alt)
                replaced = True
                break
        if not replaced:
            text = text.rstrip(".") + f", according to {rng.choice(pool)}'s filing."

    if kind == "fabricated_detail":
        text = text.rstrip() + rng.choice(FABRICATIONS)

    out = Claim.new(agent=claim.agent, section=claim.section, text=text,
                    cited_chunk_ids=list(claim.cited_chunk_ids))
    return out, kind


def build_benchmark(claims: List[Claim], seed: int = 7, owner_of=None
                    ) -> List[Dict]:
    """Produce a labelled benchmark: every genuine claim plus one perturbation.

    `owner_of(claim) -> ticker` tells the generator which company a claim is
    really about, so entity swaps are always genuine errors.
    """
    rng = random.Random(seed)
    items: List[Dict] = []
    for c in claims:
        items.append({"claim": c, "gold": "supported", "kind": "genuine"})
        kind = PERTURBATIONS[rng.randrange(len(PERTURBATIONS))]
        pc, applied = perturb(c, kind, rng, owner=owner_of(c) if owner_of else "")
        items.append({"claim": pc, "gold": "unsupported", "kind": applied})
    rng.shuffle(items)
    return items
