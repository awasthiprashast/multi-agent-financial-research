"""
agents/auditor.py — audit any pasted text, sentence by sentence.

The research pipeline verifies claims its own agents wrote. The auditor points the
same Verification Agent at text written by someone else: a broker note, a news
summary, a chatbot answer. Each sentence is placed in exactly one bucket:

  verified      every number matched real data (same value, same label) and/or the
                directional statement agrees with indicators recomputed from raw prices
  contradicted  a number the data does not support, a direction the prices contradict,
                or the sentence attributes the facts to the wrong company
  unsupported   checkable in principle, but nothing in the data backs it up
  uncheckable   no number or indicator statement, a forecast or opinion, or it compares
                several companies. It is reported, never silently "passed".

"verified" is never awarded to a sentence with nothing checkable in it.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from agents.data_collector import DataCollectionAgent
from agents.validator import (
    VerificationAgent,
    _bound,
    _claim_labels,
    _directed_value,
    _evidence_labels,
    _tokens,
)
from core import Claim, NumericFact, extract_numbers
from data.universe import BY_TICKER, UnknownCompanyError, find_tickers

MAX_CHARS = 8000
MAX_SENTENCES = 60

VERIFIED, CONTRADICTED, UNSUPPORTED, UNCHECKABLE = (
    "verified", "contradicted", "unsupported", "uncheckable")

_ABBREV = ("Rs.", "Ltd.", "Inc.", "vs.", "approx.", "Cr.", "Co.", "Corp.", "No.", "e.g.", "i.e.")
_BULLET = re.compile(r"^\s*(?:[-*•–]|\d{1,2}[.)])\s+")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹\"“(\[])")

# Forecasts, targets and recommendations cannot be checked against past data.
_OPINION = re.compile(
    r"\b(will|would|could|may|might|should|likely|expected to|expects?|forecasts?|projected|"
    r"projects?|target price|price target|outlook|recommend\w*|we believe|upside|downside|"
    r"buy|sell|overweight|underweight|attractive|undervalued|overvalued)\b", re.I)

# "FY25", "Q2", "H1", "March 31": labels for a period, not quantities.
_PERIOD_BEFORE = re.compile(
    r"(?:\bFY|\bCY|\bQ|\bH|\bpage\s|\bp\.|\bJan(?:uary)?\s|\bFeb(?:ruary)?\s|\bMar(?:ch)?\s|"
    r"\bApr(?:il)?\s|\bMay\s|\bJun(?:e)?\s|\bJul(?:y)?\s|\bAug(?:ust)?\s|\bSep(?:t(?:ember)?)?\s|"
    r"\bOct(?:ober)?\s|\bNov(?:ember)?\s|\bDec(?:ember)?\s)['’]?$", re.I)
_PERIOD_AFTER = re.compile(r"^\s*(?:st|nd|rd|th)\b", re.I)
# "50-day", "14 days", "52-week": the length of a window, not a claim about the data
_WINDOW_AFTER = re.compile(r"^[-\s]?(?:day|week|month|year|session|trading)s?\b", re.I)


@dataclass
class NumberCheck:
    raw: str
    matched: bool
    actual: Optional[str] = None       # closest value the data does hold, if any
    actual_context: Optional[str] = None


@dataclass
class Evidence:
    source: str
    doc_type: str
    published: str
    text: str


@dataclass
class AuditedSentence:
    index: int
    text: str
    ticker: Optional[str]
    status: str
    reason: str = ""
    numbers: List[NumberCheck] = field(default_factory=list)
    directions_confirmed: List[str] = field(default_factory=list)
    directions_contradicted: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    explanation: str = ""                # plain-English reason for the verdict
    correction: Optional[str] = None     # the sentence rewritten with the data's values

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AuditResult:
    sentences: List[AuditedSentence]
    tickers: List[str]
    truncated: bool = False

    @property
    def counts(self) -> Dict[str, int]:
        c = Counter(s.status for s in self.sentences)
        return {k: c.get(k, 0) for k in (VERIFIED, CONTRADICTED, UNSUPPORTED, UNCHECKABLE)}

    @property
    def checkable(self) -> int:
        return len(self.sentences) - self.counts[UNCHECKABLE]

    @property
    def accuracy(self) -> Optional[float]:
        """Verified share of the sentences that could be checked; None if none could."""
        return round(self.counts[VERIFIED] / self.checkable, 3) if self.checkable else None

    @property
    def coverage(self) -> float:
        return round(self.checkable / len(self.sentences), 3) if self.sentences else 0.0

    @property
    def headline(self) -> str:
        c = self.counts
        if not self.checkable:
            return ("None of these sentences contains a number or price statement the data "
                    "can check, so nothing could be confirmed or disproved.")
        if c[CONTRADICTED] == 0 and c[UNSUPPORTED] == 0:
            return f"All {self.checkable} checkable statements match the data."
        parts = [f"{c[VERIFIED]} of {self.checkable} checkable statements match the data"]
        if c[CONTRADICTED]:
            parts.append(f"{c[CONTRADICTED]} contradict it")
        if c[UNSUPPORTED]:
            parts.append(f"{c[UNSUPPORTED]} could not be confirmed")
        return "; ".join(parts) + "."

    @property
    def corrected_text(self) -> str:
        """The text with every wrong figure or direction replaced by what the data says."""
        return " ".join(s.correction or s.text for s in self.sentences)

    def to_dict(self) -> dict:
        return {"tickers": self.tickers, "counts": self.counts, "checkable": self.checkable,
                "accuracy": self.accuracy, "coverage": self.coverage,
                "headline": self.headline, "corrected_text": self.corrected_text,
                "truncated": self.truncated,
                "sentences": [s.to_dict() for s in self.sentences]}


# ---------------------------------------------------------------- splitting
_DASHES = re.compile("[‐-―]")                 # non-breaking hyphen, en/em dash ...
_CITE_MARKS = re.compile(r"【[^】]*】|\[\d{1,3}\]")   # 【6】 and [3] left by chatbots


def normalise(text: str) -> str:
    """Make chatbot typography comparable: "20‑day" (non-breaking hyphen) must read as
    "20-day", and citation markers such as 【6】 or [3] are not numbers."""
    text = _CITE_MARKS.sub("", text)
    text = text.replace(" ", " ").replace(" ", " ").replace(" ", " ").replace("’", "'").replace("‘", "'")
    return _DASHES.sub("-", text)


def split_sentences(text: str) -> List[str]:
    """Sentences of `text`: bullets and line breaks end a sentence, decimals and
    abbreviations such as "Rs. 100" do not."""
    out: List[str] = []
    for line in text.replace("\r", "\n").split("\n"):
        line = _BULLET.sub("", line).strip()
        if not line:
            continue
        for a in _ABBREV:
            line = line.replace(a, a.replace(".", "\x00"))
        for part in _BOUNDARY.split(line):
            part = part.replace("\x00", ".").strip()
            if len(part) >= 3:
                out.append(part)
    return out


# ------------------------------------------------------------------ helpers
_FLIP = (("above", "below"), ("higher than", "lower than"), ("over", "under"),
         ("increased", "decreased"), ("rose", "fell"), ("positive", "negative"),
         ("overbought", "oversold"))


def _plain_direction(desc: str) -> str:
    """"close 1153.8 vs sma_20 1057.81" -> a sentence a reader can follow."""
    m = re.match(r"close ([\d.]+) vs sma_(\d+) ([\d.]+)", desc)
    if m:
        c, n, a = float(m.group(1)), m.group(2), float(m.group(3))
        return (f"the last close (₹{c:,.2f}) is {'above' if c > a else 'below'} the "
                f"{n}-day average (₹{a:,.2f})")
    m = re.match(r"MACD ([-\d.]+) vs signal ([-\d.]+)", desc)
    if m:
        x, y = float(m.group(1)), float(m.group(2))
        return f"the MACD line ({x}) is {'above' if x > y else 'below'} its signal line ({y})"
    m = re.match(r"RSI ([\d.]+) is (\w+)", desc)
    if m:
        return f"the RSI is {m.group(1)}, which is in the {m.group(2)} zone"
    return desc


def _flips(sentence: str) -> List[str]:
    """Every way of reversing one direction word in the sentence."""
    out = []
    for a, b in _FLIP:
        for x, y in ((a, b), (b, a)):
            pat = re.compile("(?<![a-z])" + re.escape(x) + "(?![a-z])", re.I)
            if pat.search(sentence):
                out.append(pat.sub(y, sentence, count=1))
    return out


def _explain(s: AuditedSentence, holds) -> None:
    """Fill in the plain-English explanation and, where one exists that the data
    confirms, a corrected sentence. `holds(text)` re-checks a rewritten sentence."""
    if s.status == UNCHECKABLE:
        s.explanation = s.reason[:1].upper() + s.reason[1:] + "."
        return
    parts: List[str] = []
    fixed = s.text
    for n in s.numbers:
        if n.matched:
            continue
        if n.actual:
            parts.append(f"The data says {n.actual}, not {n.raw}.")
            fixed = fixed.replace(n.raw, n.actual, 1)
        else:
            parts.append(f"{n.raw} does not appear in the data.")
    for d in s.directions_contradicted:
        parts.append(f"Wrong: {_plain_direction(d)}.")
    if s.status == VERIFIED:
        bits = [f"{n.raw} matches the data." for n in s.numbers]
        bits += [f"Correct: {_plain_direction(d)}." for d in s.directions_confirmed]
        s.explanation = " ".join(bits) or "Matches the data."
        return
    if s.status == UNSUPPORTED:
        s.explanation = (s.reason[:1].upper() + s.reason[1:] + "."
                         if s.reason else "Nothing in the data confirms or contradicts this.")
        return
    s.explanation = " ".join(parts) or (s.reason[:1].upper() + s.reason[1:] + ".")
    # offer a rewrite only if the data confirms the rewritten sentence
    for candidate in [fixed] + _flips(fixed):
        if candidate != s.text and holds(candidate):
            s.correction = candidate
            break


def _drop_period_labels(claim: Claim) -> None:
    """Remove numbers that only name a period (FY25, Q2, "March 31")."""
    keep: List[NumericFact] = []
    for f in claim.numbers:
        before = claim.text[max(0, f.start - 12):f.start]
        after = claim.text[f.end:f.end + 6]
        if not f.unit and (_PERIOD_BEFORE.search(before) or _PERIOD_AFTER.match(after)
                           or _WINDOW_AFTER.match(after)):
            continue
        keep.append(f)
    claim.numbers = keep


def _closest_actual(verifier: VerificationAgent, fact: NumericFact, claim_text: str,
                    evidence_texts: List[str]) -> Optional[NumberCheck]:
    """What the data holds for the quantity this sentence got wrong."""
    target = _directed_value(fact, claim_text)
    labels = _claim_labels(claim_text, fact)
    wide = labels | _tokens(claim_text)
    best = None          # (distance, raw, context)

    def consider(cand: NumericFact, lab: set, context: str) -> None:
        nonlocal best
        if wide and not _bound(wide, lab):
            return
        dist = abs(cand.normalised - target) / max(abs(target), 1e-9)
        if best is None or dist < best[0]:
            best = (dist, cand.raw, context)

    def scan(text: str, context: str) -> None:
        for cand in extract_numbers(text):
            if text[max(0, cand.start - 1):cand.start] == "(":
                continue                    # the period in "RSI(14)", not a value
            if cand.unit != fact.unit or _WINDOW_AFTER.match(text[cand.end:cand.end + 12]):
                continue
            consider(cand, _evidence_labels(text, cand), context)

    store = verifier.store
    if store.source_prices and store.source_ticker:
        collector = DataCollectionAgent()
        ind = collector.compute(store.source_ticker, store.source_prices)
        for ch in collector.derived_evidence(store.source_ticker, ind, store.source_prices):
            scan(ch.text, "recomputed from the raw price series")
    for text in evidence_texts:
        scan(text, text[:160])
    if best is None:
        return None
    return NumberCheck(raw=fact.raw, matched=False, actual=best[1], actual_context=best[2])


def _evidence_rows(store, text: str, k: int = 3) -> List[Evidence]:
    rows = []
    for ch, _ in store.query(text, k=k):
        rows.append(Evidence(ch.source, ch.doc_type, ch.published, ch.text))
    return rows


# -------------------------------------------------------------------- audit
def audit_text(text: str, pipeline, ticker: Optional[str] = None) -> AuditResult:
    """Audit `text` against live (or snapshot) data held by `pipeline`.

    `ticker` is the company the text is about; if omitted it is the company the text
    names most often. A sentence that names a different supported company is checked
    against that company's data instead."""
    text = normalise(text)
    truncated = len(text) > MAX_CHARS
    sentences = split_sentences(text[:MAX_CHARS])
    if len(sentences) > MAX_SENTENCES:
        sentences, truncated = sentences[:MAX_SENTENCES], True
    if not sentences:
        raise ValueError("no text to audit")

    supported = set(pipeline.supported)
    if ticker is not None and ticker not in supported:
        raise UnknownCompanyError(f"{ticker} is not supported; available: {', '.join(sorted(supported))}")
    mention_counts = Counter(t for s in sentences for t in find_tickers(s) if t in supported)
    default = ticker or (mention_counts.most_common(1)[0][0] if mention_counts else None)
    if default is None:
        raise UnknownCompanyError(
            "the text names no supported company; pick one. Supported: "
            + ", ".join(BY_TICKER[t].name for t in sorted(supported)))

    results: List[AuditedSentence] = []
    verifiers: Dict[str, VerificationAgent] = {}

    for i, sent in enumerate(sentences):
        named = [t for t in find_tickers(sent) if t in supported]
        if len(named) > 1:
            results.append(AuditedSentence(i, sent, None, UNCHECKABLE,
                                           "compares several companies; not checked"))
            continue
        tic = named[0] if named else default
        ctx = pipeline._prepare(tic)
        store = ctx["store"]
        verifier = verifiers.get(tic)
        if verifier is None:
            verifier = verifiers[tic] = VerificationAgent(store, threshold=pipeline.threshold)

        claim = Claim.new("external", "audit", sent)
        _drop_period_labels(claim)
        conflicts, confirmed = verifier._logic_check(claim)

        if _OPINION.search(sent):
            results.append(AuditedSentence(i, sent, tic, UNCHECKABLE,
                                           "forecast, target or opinion; cannot be checked against past data"))
            continue
        if not claim.numbers and not conflicts and not confirmed:
            results.append(AuditedSentence(i, sent, tic, UNCHECKABLE,
                                           "no number or indicator statement to check"))
            continue

        verdict = verifier.verify(claim)
        evidence = _evidence_rows(store, sent)
        evidence_texts = [e.text for e in evidence]

        checks: List[NumberCheck] = []
        for f in claim.numbers:
            if f.raw in verdict.unsupported_numbers:
                checks.append(_closest_actual(verifier, f, sent, evidence_texts)
                              or NumberCheck(raw=f.raw, matched=False))
            else:
                checks.append(NumberCheck(raw=f.raw, matched=True))

        reason = verdict.reason
        has_comparator = any(not n.matched and n.actual for n in checks)
        if conflicts or reason.startswith("claim entity") or has_comparator:
            # the data holds a comparable value and it differs, or the direction is wrong
            status = CONTRADICTED
        elif verdict.unsupported_numbers:
            # the data holds nothing to compare this figure with: not proven wrong
            status = UNSUPPORTED
            reason = "no figure in the data to compare with: " + ", ".join(verdict.unsupported_numbers)
        elif (claim.numbers and verdict.numeric_support >= 1.0) or (confirmed and not conflicts):
            status, reason = VERIFIED, ""
        else:
            status = UNSUPPORTED
            reason = reason or "nothing in the data backs this up"

        results.append(AuditedSentence(
            i, sent, tic, status, reason, checks, list(confirmed), list(conflicts), evidence))

    def holds_for(tic: str):
        verifier = verifiers[tic]

        def holds(text: str) -> bool:
            c = Claim.new("external", "audit", text)
            _drop_period_labels(c)
            conflicts, confirmed = verifier._logic_check(c)
            v = verifier.verify(c)
            return not v.unsupported_numbers and not conflicts and (
                (bool(c.numbers) and v.numeric_support >= 1.0) or bool(confirmed))
        return holds

    for s in results:
        _explain(s, holds_for(s.ticker) if s.ticker in verifiers else (lambda _t: False))
    return AuditResult(results, sorted({s.ticker for s in results if s.ticker}), truncated)
