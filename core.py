"""
core.py — shared data structures for the Multi-Agent Financial Research Assistant.

Every analytical statement produced anywhere in the system is a `Claim`. A Claim
is never allowed to reach the user without passing through the Verification Agent,
which attaches a `Verdict`. This is the structural guarantee the project rests on.
"""

from __future__ import annotations

import hashlib
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------

@dataclass
class Chunk:
    """A retrievable unit of source evidence with full provenance."""
    chunk_id: str
    text: str
    source: str            # e.g. "AR2025_TCS_p42", "moneycontrol_rss"
    doc_type: str          # filing | news | market_data | macro
    ticker: str
    published: str         # ISO date — 'what was known when'
    meta: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(source: str, text: str) -> str:
        h = hashlib.sha1((source + "||" + text).encode("utf-8")).hexdigest()[:12]
        return f"{source}#{h}"


# --------------------------------------------------------------------------
# Claims
# --------------------------------------------------------------------------

# Numeric token: 12.4 | 12.4% | ₹1,234 | 1,234.5 cr | -3.2% | 1.8x
#
# The grouped-digits alternative requires at least one comma group and comes
# first, so "₹240,893" is read as one quantity. The plain alternative is anchored
# on both sides so a bare "2026" cannot be split into "202" and "6" — a real bug
# this regex had until the test suite caught it.
NUM_RE = re.compile(
    r"(?P<sign>[-+−]?)\s*(?P<cur>₹|Rs\.?|\$)?\s*"
    r"(?P<num>\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|(?<![\d.,])\d+(?:\.\d+)?(?!\d|,\d))\s*"
    r"(?P<unit>%|percent|per cent|bps|x|cr|crore|lakh|bn|billion|mn|million)?",
    re.IGNORECASE,
)

_SCALE = {
    "": 1.0, "%": 1.0, "percent": 1.0, "per cent": 1.0, "x": 1.0,
    "bps": 0.01,
    "lakh": 1e5, "cr": 1e7, "crore": 1e7,
    "mn": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9,
}


@dataclass
class NumericFact:
    """A number lifted out of free text, normalised so it can be compared."""
    raw: str
    value: float
    unit: str
    start: int = -1        # offset of the digits in the source text
    end: int = -1          # offset just past the number and its unit

    @property
    def normalised(self) -> float:
        return self.value * _SCALE.get(self.unit.lower(), 1.0)


def extract_numbers(text: str) -> List[NumericFact]:
    """Pull every comparable number out of a piece of text.

    This is deliberately not an LLM call. Numeric hallucination is the failure
    mode that matters most in finance (FinQA), and a regex + normalisation pass
    is deterministic, auditable and free — which is the whole point.
    """
    facts: List[NumericFact] = []
    for m in NUM_RE.finditer(text):
        raw_num = m.group("num")
        if raw_num is None:
            continue
        try:
            val = float(raw_num.replace(",", ""))
        except ValueError:
            continue
        if m.group("sign") in ("-", "−"):
            val = -val
        unit = (m.group("unit") or "").strip()
        # Skip bare years and list indices — they are not analytical quantities.
        if not unit and 1900 <= val <= 2100 and float(val).is_integer():
            continue
        facts.append(NumericFact(raw=m.group(0).strip(), value=val, unit=unit,
                                 start=m.start("num"), end=m.end()))
    return facts


@dataclass
class Claim:
    """One analytical statement made by one agent, with its asserted evidence."""
    claim_id: str
    agent: str
    section: str
    text: str
    cited_chunk_ids: List[str] = field(default_factory=list)
    numbers: List[NumericFact] = field(default_factory=list)
    revision: int = 0

    @classmethod
    def new(cls, agent: str, section: str, text: str,
            cited_chunk_ids: Optional[List[str]] = None) -> "Claim":
        return cls(
            claim_id=uuid.uuid4().hex[:10],
            agent=agent,
            section=section,
            text=text,
            cited_chunk_ids=list(cited_chunk_ids or []),
            numbers=extract_numbers(text),
        )


# --------------------------------------------------------------------------
# Verification output
# --------------------------------------------------------------------------

@dataclass
class Verdict:
    """The Verification Agent's judgement on a single claim."""
    claim_id: str
    confidence: float
    numeric_support: float
    semantic_support: float
    citation_consistent: bool
    supporting_chunk_ids: List[str] = field(default_factory=list)
    unsupported_numbers: List[str] = field(default_factory=list)
    status: str = "verified"          # verified | revised | flagged
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Report:
    query: str
    ticker: str
    sections: Dict[str, List[Claim]]
    verdicts: Dict[str, Verdict]
    agents_invoked: List[str]
    timings: Dict[str, float]
    generated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    loop_stats: Dict[str, int] = field(default_factory=dict)
    mode: str = "unknown"                 # "live" | "snapshot"
    validator_calls: int = 0
    store: Any = None                     # EvidenceStore, for rendering sources
    withheld: List[Any] = field(default_factory=list)     # (Claim, Verdict) never surfaced
    first_pass: List[Any] = field(default_factory=list)   # (Claim, Verdict) before revision
    analysts: str = "template"            # "template" | "llm"

    @property
    def surfaced_count(self) -> int:
        """Claims that reached the user: verified outright or after revision."""
        return sum(1 for v in self.verdicts.values()
                   if v.status in ("verified", "revised"))

    @property
    def grounding_rate(self) -> float:
        """Surfaced claims / claims generated. One definition, used everywhere."""
        if not self.verdicts:
            return 0.0
        return self.surfaced_count / len(self.verdicts)
