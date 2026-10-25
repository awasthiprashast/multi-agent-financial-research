"""
agents/validator.py — the Verification Agent.

This is the contribution of the project. Everything else is assembly; this is the
part that is novel, measurable, and the subject of the paper and provisional
patent draft.

The problem: retrieval-augmented generation grounds the *input* to a generator.
Nothing re-checks the *output*. Once a claim is written, no component in a
standard RAG or multi-agent stack asks whether that specific sentence — and
specifically the numbers inside it — is actually supported by the passage it
purports to come from. Self-consistency methods (SelfCheckGPT) sample the model
repeatedly and measure agreement, which is expensive and is a test of stability,
not of truth. Agent-debate methods (AutoGen, Reflexion) have models check each
other, which is a test of consensus, not of evidence.

The mechanism implemented here, stated as a procedure:

  1. INDEPENDENT RE-RETRIEVAL. Take the claim text. Embed it. Retrieve the top-k
     evidence chunks from the store, ignoring the citation the agent attached.
     This makes the citation falsifiable: if the agent's cited chunk does not
     come back, the citation is not corroborated.

  2. NUMERIC ADJUDICATION. Extract every quantity from the claim and normalise
     it for scale and unit (%, bps, cr, x, ₹). For each quantity, search the
     union of the cited chunk and the re-retrieved chunks for a value that
     matches within a relative tolerance τ. numeric_support is the fraction of
     the claim's quantities that are matched. A claim with an unmatched number
     is treated as unsupported no matter how fluent it reads — because in
     finance the number is the claim.

  3. SEMANTIC ADJUDICATION. semantic_support is the maximum cosine similarity
     between the claim and any re-retrieved chunk. This catches assertions that
     carry no numbers at all.

  4. CITATION CONSISTENCY. A boolean: did the agent's asserted source survive
     independent re-retrieval?

  5. CONFIDENCE. A weighted combination, with the weights shifted towards the
     numeric term whenever the claim contains numbers:

        confidence = w_n·numeric + w_s·semantic + w_c·citation

  6. BOUNDED REVISION. If confidence < θ, the claim is returned to the agent
     that produced it with the specific failing quantity named. The agent gets
     at most R attempts. If it still fails, the claim is dropped and recorded as
     flagged — it never reaches the user. Bounding the loop is what keeps cost
     finite on a free-tier deployment.

The headline metric the project reports is the fraction of surfaced claims that
are traceably grounded, together with the fraction of deliberately injected
false claims the mechanism rejects.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from agents.data_collector import DataCollectionAgent
from core import NUM_RE, Claim, NumericFact, Verdict, extract_numbers
from data.universe import COMPANIES
from rag.store import EvidenceStore

# --------------------------------------------------------------------------
# Number-to-label binding. A value only counts as supported if the evidence
# number it matches sits next to the same words: "RSI is 50" must not be
# grounded by the "50" in "50-day moving average".
# --------------------------------------------------------------------------
_STOP = frozenset("""
a an the of for in on at to by and or is are was were be been being as with from
that this which it its year years over against than into per rs cr crore lakh
bps bn mn x percent cent stood stands remained reported about approximately
""".split())

_TOKEN = re.compile(r"[a-z]+")
_SENT_SPLIT = re.compile(r"(?<=\S)\.\s+(?=[A-Z0-9₹])")
_DOWN = re.compile(
    r"\b(?:fell|fall|falls|fallen|declined?|dropped?|drops?|down|decreased?|"
    r"lower|lost|contracted|slipped|dipped)\b((?:\s+(?!to\b)[\w-]+){0,3})\W*$",
    re.IGNORECASE)
_SIGNED_UNITS = {"%", "percent", "per cent", "bps"}


# Word forms that mean the same thing in a market-data sentence. Without this, "a
# high of 1493" cannot be matched to "the highest close was 1493".
_CANON = {
    "highest": "high", "higher": "high", "peak": "high", "top": "high",
    "lowest": "low", "lower": "low", "trough": "low", "bottom": "low",
    "closed": "close", "closing": "close", "closes": "close",
    "fell": "decline", "fall": "decline", "fallen": "decline", "declined": "decline",
    "drop": "decline", "dropped": "decline", "decreased": "decline",
    "decrease": "decline", "dipped": "decline",
    "rose": "rise", "risen": "rise", "gained": "rise", "increased": "rise",
    "increase": "rise", "climbed": "rise",
    "volatile": "volatility", "annualized": "annualised",
}


# Words too common to identify a quantity on their own: "operating margin" and "net
# margin" share "margin", but they are different numbers.
_GENERIC = frozenset({"margin", "ratio", "total", "rate", "level", "value", "score",
                      "index", "figure", "percentage", "amount", "number"})


def _bound(claim_labels: set, evidence_labels: set) -> bool:
    """Do the two describe the same quantity? Distinguishing words must overlap when
    the claim has any; only if it has none is a generic word enough."""
    strong = claim_labels - _GENERIC
    return bool(strong & evidence_labels) if strong else bool(claim_labels & evidence_labels)


def _tokens(text: str) -> set:
    """Content words of `text`, numbers removed, lightly stemmed and canonicalised."""
    out = set()
    for w in _TOKEN.findall(NUM_RE.sub(" ", text).lower()):
        if w in _STOP or len(w) < 2:
            continue
        w = _CANON.get(w, w)
        out.add(w[:-1] if len(w) > 3 and w.endswith("s") else w)
    return out


_CLAUSE = re.compile(r",\s|;|:|\band\b|\bwhile\b|\bbut\b|\bwith\b|\bwhich\b|"
                     r"\bindicating\b|\bsuggest(?:ing|s)?\b", re.IGNORECASE)


def _claim_labels(text: str, fact: NumericFact, before: int = 8, after: int = 3) -> set:
    """Words that describe a number inside the claim sentence.

    The union of (a) a short window around the number and (b) the whole clause it
    sits in. The clause matters for long LLM sentences where the subject
    ("annualised volatility ... for <Company> over the given period") is far from
    the number; the window matters for compact ones ("up 6.8% year on year").
    """
    if fact.start < 0:
        return set()
    pre = [w for w in _TOKEN.findall(NUM_RE.sub(" ", text[:fact.start]).lower())]
    post = [w for w in _TOKEN.findall(NUM_RE.sub(" ", text[fact.end:]).lower())]
    window = _tokens(" ".join(pre[-before:] + post[:after]))
    lo = max([0] + [m.end() for m in _CLAUSE.finditer(text) if m.end() <= fact.start])
    hi = min([len(text)] + [m.start() for m in _CLAUSE.finditer(text) if m.start() >= fact.end])
    return window | _tokens(text[lo:hi])


def _evidence_labels(text: str, cand: NumericFact) -> set:
    """Words in the evidence sentence that contains `cand`."""
    pos = 0
    for part in _SENT_SPLIT.split(text):
        idx = text.find(part, pos)
        end = idx + len(part)
        if idx <= cand.start < end + 1:
            return _tokens(part)
        pos = end
    return _tokens(text)


# Quantities customarily quoted as a magnitude although they are negative
# ("drawdown of 25.7%", "a one-day VaR loss of 2.6%"): sign is not compared.
_MAGNITUDE_LABELS = frozenset({"drawdown", "loss", "risk"})
# Conventional RSI thresholds are definitions, not data about the company.
_RSI_THRESHOLDS = (30.0, 70.0)


_UP = r"(?:above|over|higher than|greater than|exceeds?|exceeding)"
_DN = r"(?:below|under|lower than|less than)"
_PRICE = r"(?:close|closing price|last close|price|stock|shares)"


def _window_keys(tail: str, default: bool) -> List[str]:
    """Moving averages named in `tail` ("20-day", "200-day"); the 20/50 pair if none is."""
    keys = [k for w, k in (("20", "sma_20"), ("50", "sma_50"), ("200", "sma_200"))
            if re.search(rf"(?<!\d){w}(?!\d)", tail)]
    return keys or (["sma_20", "sma_50"] if default else [])


def _direction_conflicts(text: str, ind: dict, closes: list,
                         confirmed: Optional[List[str]] = None) -> List[str]:
    """Qualitative statements the recomputed indicators contradict.

    Numbers are not the only way to be wrong: "MACD is below its signal line"
    carries no quantity at all. These rules cover the directional claims the
    technical agents make, checked against values recomputed from raw prices.
    Statements the recomputation *confirms* are appended to `confirmed`, so a
    numberless claim that is verifiably true is not penalised for having no
    numbers to match.
    """
    # "INFY.NS" must not look like a sentence boundary to the [^.;] spans below;
    # decimals ("23.8") are untouched because a digit follows their dot.
    low = re.sub(r"(?<=[a-z0-9])\.(?=[a-z])", "", text.lower())
    # ...and a decimal point is not a sentence boundary either: "closed at ₹1512.37,
    # lower than the 50-day average" must reach the comparison. (A middle dot keeps
    # the digits apart without ending the [^.;] spans.)
    low = re.sub(r"(?<=\d)\.(?=\d)", "·", low)
    low = re.sub(r"\b(upper|lower) bollinger(?: bands?)?", r"bollinger \1 band", low)
    out: List[str] = []

    def judge(ok: bool, desc: str) -> None:
        if not ok:
            out.append(desc)
        elif confirmed is not None:
            confirmed.append(desc)

    def neg_before(pos: int) -> bool:
        return bool(re.search(r"\b(?:not|n't|neither|nor|no)\b", low[max(0, pos - 18):pos]))

    # price vs moving averages
    for m in re.finditer(rf"{_PRICE}[^.;]{{0,60}}?\b({_UP}|{_DN})\b([^.;]{{0,70}}?"
                         r"(?:moving average|sma|averages?))", low):
        if neg_before(m.start(1)):
            continue
        claims_up = bool(re.match(_UP, m.group(1)))
        tail = m.group(2)
        keys = []
        keys = _window_keys(tail, default=False)
        keys = keys or ["sma_20", "sma_50"]
        for k in keys:
            avg = ind.get(k)
            if avg is None or avg != avg:
                continue
            judge(claims_up == (ind["last_close"] > avg),
                  f"close {ind['last_close']} vs {k} {avg}")

    def vs_average(claims_up: bool, keys: List[str]) -> None:
        for k in keys:
            avg = ind.get(k)
            if avg is not None and avg == avg:
                judge(claims_up == (ind["last_close"] > avg),
                      f"close {ind['last_close']} vs {k} {avg}")

    def window_keys(tail: str) -> List[str]:
        return _window_keys(tail, default=True)

    # continuation clause: "... is higher than the 20-day average but lower than the
    # 50-day average" — the subject (the price) is carried over from the first clause
    for m in re.finditer(rf"\b(?:but|while|whereas|and)\s+(?:is\s+|remains\s+)?({_UP}|{_DN})\b"
                         r"\s+(?:the\s+|its\s+)?(\d+)-day[^.;]{0,25}?(?:moving average|sma|average)",
                         low):
        if neg_before(m.start(1)) or not re.search(rf"\b{_PRICE}\b", low[:m.start()]):
            continue
        vs_average(bool(re.match(_UP, m.group(1))), window_keys(m.group(2)))

    # average as the subject: "the 20-day moving average is lower than the closing price"
    for m in re.finditer(rf"((?:\d+-day\s+)?(?:simple\s+)?(?:moving average|sma|average)s?)"
                         rf"[^.;]{{0,40}}?\b({_UP}|{_DN})\b[^.;]{{0,25}}?"
                         r"\b(?:close|closing price|price)\b", low):
        if neg_before(m.start(2)):
            continue
        # the average is higher than the price  <=>  the price is NOT above the average
        vs_average(not bool(re.match(_UP, m.group(2))), window_keys(m.group(1)))

    # equality: "the close is equal to the 20-day average" — true only within τ
    for m in re.finditer(rf"\b{_PRICE}\b[^.;]{{0,50}}?\b(?:equal to|equals|same as|identical to)\b"
                         r"([^.;]{0,50}?(?:moving average|sma|average))", low):
        for k in window_keys(m.group(1)):
            avg = ind.get(k)
            if avg is not None and avg == avg and avg != 0:
                judge(abs(ind["last_close"] - avg) / abs(avg) <= 0.005,
                      f"close {ind['last_close']} is not equal to {k} {avg}")

    # MACD line vs signal line
    for m in re.finditer(rf"macd[^.;]{{0,40}}?\b({_UP}|{_DN})\b[^.;]{{0,25}}?signal", low):
        if neg_before(m.start(1)):
            continue
        claims_up = bool(re.match(_UP, m.group(1)))
        judge(claims_up == (ind["macd_line"] > ind["macd_signal"]),
              f"MACD {ind['macd_line']} vs signal {ind['macd_signal']}")

    # histogram sign
    for m in re.finditer(r"histogram[^.;]{0,25}?\b(positive|negative)\b", low):
        if neg_before(m.start(1)):
            continue
        if ind["macd_hist"] != 0:
            judge((m.group(1) == "positive") == (ind["macd_hist"] > 0),
                  f"MACD histogram {ind['macd_hist']}")

    # RSI zone
    for m in re.finditer(r"rsi[^.;]{0,70}?\b(overbought|oversold|neutral)\b", low):
        if neg_before(m.start(1)):
            continue
        r = ind["rsi_14"]
        zone = "overbought" if r >= 70 else "oversold" if r <= 30 else "neutral"
        judge(m.group(1) == zone, f"RSI {r} is {zone}")

    # RSI against a stated threshold: "RSI is above 70"
    for m in re.finditer(rf"rsi[^.;]{{0,70}}?\b({_UP}|{_DN})\b[^.;]{{0,15}}?\b(\d{{2,3}})\b", low):
        if neg_before(m.start(1)):
            continue
        thr = float(m.group(2))
        judge(bool(re.match(_UP, m.group(1))) == (ind["rsi_14"] > thr),
              f"RSI {ind['rsi_14']} vs {thr:g}")

    # Bollinger band vs price: "the upper band is higher than the closing price",
    # "price is below the lower band"
    for m in re.finditer(rf"bollinger[^.;]{{0,30}}?\b(upper|lower)\b[^.;]{{0,40}}?"
                         rf"\b({_UP}|{_DN})\b[^.;]{{0,25}}?\b(?:close|closing price|price)\b", low):
        if neg_before(m.start(2)):
            continue
        band = ind["bb_upper"] if m.group(1) == "upper" else ind["bb_lower"]
        judge(bool(re.match(_UP, m.group(2))) == (band > ind["last_close"]),
              f"{m.group(1)} band {band} vs close {ind['last_close']}")
    for m in re.finditer(rf"\b(?:close|closing price|price)\b[^.;]{{0,40}}?\b({_UP}|{_DN})\b"
                         r"[^.;]{0,25}?\b(upper|lower)\b[^.;]{0,15}?(?:bollinger|band)", low):
        if neg_before(m.start(1)):
            continue
        band = ind["bb_upper"] if m.group(2) == "upper" else ind["bb_lower"]
        judge(bool(re.match(_UP, m.group(1))) == (ind["last_close"] > band),
              f"close {ind['last_close']} vs {m.group(2)} band {band}")

    # extremes of the window: "the last close is the lowest close in the window"
    if closes:
        for m in re.finditer(r"\b(?:last|closing|current)\b[^.;]{0,50}?\b(highest|lowest)\b", low):
            if neg_before(m.start(1)):
                continue
            target = max(closes) if m.group(1) == "highest" else min(closes)
            judge(closes[-1] == target, f"last close {closes[-1]} vs {m.group(1)} {target}")

    # "the price has decreased / increased over the window"
    for m in re.finditer(r"\b(?:close|closing price|price|stock|shares)\b[^.;]{0,30}?"
                         r"\b(decreased|declined|fell|dropped|increased|rose|risen|gained)\b"
                         r"[^.;]{0,35}?\b(?:sessions|window|period|year)\b", low):
        if neg_before(m.start(1)):
            continue
        went_up = m.group(1) in ("increased", "rose", "risen", "gained")
        if ind["period_return_pct"] != 0:
            judge(went_up == (ind["period_return_pct"] > 0),
                  f"period return {ind['period_return_pct']}%")

    # return over the window
    ret = ind["period_return_pct"]
    for m in re.finditer(r"\b(positive|negative)\b[^.;]{0,12}?(?:price )?return", low):
        if neg_before(m.start(1)):
            continue
        if ret != 0:
            judge((m.group(1) == "positive") == (ret > 0), f"period return {ret}%")
    if closes:
        for m in re.finditer(rf"(?:last|closing)[^.;]{{0,45}}?\b({_UP}|{_DN})\b[^.;]{{0,20}}?"
                             r"(?:starting|first|opening|initial) price", low):
            if neg_before(m.start(1)):
                continue
            judge(bool(re.match(_UP, m.group(1))) == (closes[-1] > closes[0]),
                  f"last close {closes[-1]} vs start {closes[0]}")
    return out


def _directed_value(fact: NumericFact, claim_text: str) -> float:
    """Normalised value with the direction the claim's wording implies.

    "margin fell 70 bps" asserts -70 bps even though no minus sign is printed.
    "declined to 24.3%" does not: the number after "to" is a level.
    """
    v = fact.normalised
    if fact.unit.lower() in _SIGNED_UNITS and v > 0 and fact.start >= 0:
        window = claim_text[max(0, fact.start - 45):fact.start]
        if _DOWN.search(window):
            return -v
    return v


class VerificationAgent:
    name = "validator"

    def __init__(
        self,
        store: EvidenceStore,
        k: int = 5,
        tolerance: float = 0.005,       # τ — relative tolerance for numeric match
        threshold: float = 0.60,        # θ — confidence floor for surfacing
        max_retries: int = 2,           # R — bound on the revision loop
        w_numeric: float = 0.55,
        w_semantic: float = 0.30,
        w_citation: float = 0.15,
        numeric_veto: bool = True,      # ablation switch — see below
        entity_check: bool = True,      # ablation switch — see below
        logic_check: bool = True,       # directional claims vs recomputed indicators
        veto_ceiling: float = 0.45,
    ):
        self.store = store
        self.k = k
        self.tolerance = tolerance
        self.threshold = threshold
        self.max_retries = max_retries
        self.w_numeric = w_numeric
        self.w_semantic = w_semantic
        self.w_citation = w_citation
        self.numeric_veto = numeric_veto
        self.entity_check = entity_check
        self.logic_check = logic_check
        self.veto_ceiling = veto_ceiling
        self.calls = 0
        self.first_pass: list = []
        self.withheld: list = []

    # ------------------------------------------------------------ numeric
    def _close(self, c: float, target: float) -> bool:
        if target == 0:
            return abs(c) <= 1e-9
        return abs(c - target) / max(abs(target), 1e-9) <= self.tolerance

    def _recomputed(self) -> list:
        """(number, label words) for every indicator, recomputed now from the raw
        price series. Cached on the store: the series never changes under it."""
        st = self.store
        if st._recompute_cache is None:
            facts: list = []
            if st.source_prices and st.source_ticker:
                agent = DataCollectionAgent()
                ind = agent.compute(st.source_ticker, st.source_prices)
                for ch in agent.derived_evidence(st.source_ticker, ind, st.source_prices):
                    for cand in extract_numbers(ch.text):
                        # the 14 in "RSI(14)" is a period, not a value: only a claim that
                        # writes it the same way, "RSI(14)", may be grounded by it
                        cand.paren = ch.text[max(0, cand.start - 1):cand.start] == "("
                        facts.append((cand, _evidence_labels(ch.text, cand)))
                st._recompute_ind = ind
            st._recompute_cache = facts
        return st._recompute_cache

    def _logic_check(self, claim: Claim):
        """(contradicted, confirmed) directional statements in the claim."""
        self._recomputed()
        ind = getattr(self.store, "_recompute_ind", None)
        if not ind:
            return [], []
        closes = [float(c) for c in (self.store.source_prices or {}).get("close", [])]
        confirmed: List[str] = []
        conflicts = _direction_conflicts(claim.text, ind, closes, confirmed)
        return conflicts, confirmed

    def _numeric_match(self, fact: NumericFact, claim_text: str,
                       evidence_texts: List[str]) -> bool:
        """True if some evidence number equals the claim's within τ *and* is
        described by at least one of the same words (label binding).

        Numbers about computed indicators are adjudicated against a recomputation
        from raw prices when one is available; everything else against text."""
        target = _directed_value(fact, claim_text)
        labels = _claim_labels(claim_text, fact)

        if fact.unit == "" and abs(fact.value) in _RSI_THRESHOLDS \
                and labels & {"rsi", "overbought", "oversold"}:
            return True
        magnitude = bool(labels & _MAGNITUDE_LABELS) and fact.raw.lstrip()[:1] not in "-+−"

        def close(c: float) -> bool:
            return self._close(c, target) or (magnitude and self._close(abs(c), abs(target)))

        # recomputed from raw prices: always available, whatever retrieval returned.
        # The whole claim sentence may name the indicator ("The RSI ... indicates
        # oversold conditions with a value of 23.8"), so labels are matched wide.
        wide = labels | _tokens(claim_text)
        fact_paren = claim_text[max(0, fact.start - 1):fact.start] == "("
        for cand, lab in self._recomputed():
            if getattr(cand, "paren", False) and not fact_paren:
                continue
            if close(cand.normalised) and (not wide or _bound(wide, lab)):
                return True

        for text in evidence_texts:
            for cand in extract_numbers(text):
                c = cand.normalised
                if not close(c):
                    continue
                if labels and not _bound(labels, _evidence_labels(text, cand)):
                    continue
                return True
        return False

    # ------------------------------------------------------------- entity
    #: Company surface forms the system knows about. Extending this list is the
    #: only change needed to cover a new issuer.
    ENTITY_FORMS = {c.ticker: list(c.aliases) for c in COMPANIES}
    ENTITY_FORMS["HDFCBANK.NS"].append("hdfc")      # the bare group name also names the bank

    def _entities_in(self, text: str) -> set:
        # Whole-word match: "ril" must not fire inside "April" or "brilliant".
        low = text.lower()
        found = set()
        for tic, forms in self.ENTITY_FORMS.items():
            if any(re.search(rf"(?<![a-z0-9]){re.escape(f)}(?![a-z0-9])", low)
                   for f in forms):
                found.add(tic)
        return found

    def _entity_consistent(self, claim: Claim, hits) -> Tuple[bool, str]:
        claimed = self._entities_in(claim.text)
        if not claimed:
            return True, ""            # no entity asserted, nothing to contradict
        evidence_entities = set()
        for chunk, _ in hits:
            evidence_entities |= self._entities_in(chunk.text)
            if chunk.ticker:
                evidence_entities.add(chunk.ticker)
        for cid in claim.cited_chunk_ids:
            ch = self.store.get(cid)
            if ch is not None:
                evidence_entities |= self._entities_in(ch.text)
                evidence_entities.add(ch.ticker)
        unsupported_entities = claimed - evidence_entities
        if unsupported_entities:
            return False, ", ".join(sorted(unsupported_entities))
        return True, ""

    # ------------------------------------------------------------- verify
    def verify(self, claim: Claim) -> Verdict:
        self.calls += 1
        hits: List[Tuple] = self.store.query(claim.text, k=self.k)
        retrieved_ids = [c.chunk_id for c, _ in hits]
        retrieved_texts = [c.text for c, _ in hits]
        top_sim = max((s for _, s in hits), default=0.0)

        cited_texts = []
        for cid in claim.cited_chunk_ids:
            ch = self.store.get(cid)
            if ch is not None:
                cited_texts.append(ch.text)

        # Indexed computed-indicator chunks are NOT trusted as numeric evidence when
        # the raw series is available: those numbers are recomputed instead.
        if self._recomputed():
            keep = lambda ch: not (ch.doc_type == "market_data"
                                   and ch.meta.get("provenance") == "computed")
            retrieved_texts = [c.text for c, _ in hits if keep(c)]
            cited_texts = [self.store.get(cid).text for cid in claim.cited_chunk_ids
                           if self.store.get(cid) is not None and keep(self.store.get(cid))]
        evidence = retrieved_texts + cited_texts

        # (2) numeric adjudication
        unsupported: List[str] = []
        if claim.numbers:
            matched = 0
            for fact in claim.numbers:
                if self._numeric_match(fact, claim.text, evidence):
                    matched += 1
                else:
                    unsupported.append(fact.raw)
            numeric_support = matched / len(claim.numbers)
        else:
            numeric_support = 1.0            # nothing numeric to contradict

        # (3) semantic adjudication
        semantic_support = max(0.0, min(1.0, float(top_sim)))

        # (4) citation consistency
        citation_ok = bool(claim.cited_chunk_ids) and any(
            cid in retrieved_ids for cid in claim.cited_chunk_ids)

        # (3b) directional statements checked against the recomputed indicators. A
        # claim whose every checkable statement is *confirmed* is supported by the
        # raw data itself, so its lack of numbers (semantic overlap) or of a
        # document citation does not count against it.
        conflicts, confirmed = self._logic_check(claim) if self.logic_check else ([], [])
        by_recomputation = bool(confirmed) and not conflicts
        if by_recomputation:
            semantic_support = max(semantic_support, 0.85)
        cite_term = citation_ok or by_recomputation

        # (4b) entity binding — a claim naming company X may not be grounded in
        # evidence about company Y. Without this check a perfectly accurate
        # sentence transplanted onto the wrong issuer passes every other test,
        # which is precisely the failure a reader would never catch by eye.
        entity_ok = True
        entity_conflict = ""
        if self.entity_check:
            entity_ok, entity_conflict = self._entity_consistent(claim, hits)

        # (5) confidence — weight numerics harder when the claim contains them
        if claim.numbers:
            wn, ws, wc = self.w_numeric, self.w_semantic, self.w_citation
        else:
            wn, ws, wc = 0.0, self.w_semantic + self.w_numeric * 0.6, \
                self.w_citation + self.w_numeric * 0.4
        confidence = wn * numeric_support + ws * semantic_support + wc * (1.0 if cite_term else 0.0)

        # (5b) numeric veto. In financial text a single unsupported quantity
        # invalidates the sentence regardless of how well the rest of it scores,
        # so the numeric term acts as a veto rather than as one vote among
        # several. Disabling this switch reproduces the purely additive scorer,
        # which is how the ablation in the report is produced.
        if self.numeric_veto and unsupported:
            confidence = min(confidence, self.veto_ceiling)
        if self.entity_check and not entity_ok:
            confidence = min(confidence, self.veto_ceiling)
        if conflicts:
            confidence = min(confidence, self.veto_ceiling)

        confidence = round(min(1.0, confidence), 3)

        status = "verified" if confidence >= self.threshold else "flagged"
        reason = ""
        if unsupported:
            reason = ("no matching value in retrieved evidence for: "
                      + ", ".join(unsupported))
        elif conflicts:
            reason = ("claim contradicts values recomputed from raw prices: "
                      + "; ".join(conflicts))
        elif not entity_ok:
            reason = f"claim entity not supported by retrieved evidence: {entity_conflict}"
        elif status == "flagged":
            reason = "insufficient semantic support in retrieved evidence"

        return Verdict(
            claim_id=claim.claim_id,
            confidence=confidence,
            numeric_support=round(numeric_support, 3),
            semantic_support=round(semantic_support, 3),
            citation_consistent=citation_ok,
            supporting_chunk_ids=retrieved_ids[:3],
            unsupported_numbers=unsupported,
            status=status,
            reason=reason,
        )

    # ------------------------------------------------------- revision loop
    def verify_with_revision(
        self,
        claims: List[Claim],
        agents_by_name: Dict[str, object],
        ticker: str,
        context: dict,
    ) -> Tuple[List[Claim], Dict[str, Verdict], Dict[str, int]]:
        """Run step 6. Returns (surfaced claims, verdicts, loop statistics)."""
        surfaced: List[Claim] = []
        verdicts: Dict[str, Verdict] = {}
        stats = {"revisions_attempted": 0, "revisions_successful": 0, "dropped": 0}
        self.first_pass = []        # (claim, verdict) before any revision
        self.withheld = []          # (claim, verdict) for claims that never surfaced

        for claim in claims:
            current = claim
            verdict = self.verify(current)
            self.first_pass.append((claim, verdict))

            attempts = 0
            while verdict.status == "flagged" and attempts < self.max_retries:
                agent = agents_by_name.get(current.agent)
                if agent is None:
                    break
                stats["revisions_attempted"] += 1
                revised = agent.revise(current, verdict.reason, ticker, context)
                attempts += 1
                if revised is None:
                    break
                new_verdict = self.verify(revised)
                if new_verdict.confidence > verdict.confidence:
                    current, verdict = revised, new_verdict
                if verdict.status == "verified":
                    verdict.status = "revised"
                    stats["revisions_successful"] += 1
                    break

            verdicts[current.claim_id] = verdict
            if verdict.status in ("verified", "revised"):
                surfaced.append(current)
            else:
                stats["dropped"] += 1
                self.withheld.append((current, verdict))

        return surfaced, verdicts, stats
