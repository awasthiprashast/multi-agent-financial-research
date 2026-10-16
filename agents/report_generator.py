"""
agents/report_generator.py — Report Generator Agent.

Assembles surfaced claims into a structured report. Every sentence carries its
confidence and its source ids; nothing that failed verification appears. The
disclaimer is not optional and is not configurable.
"""

from __future__ import annotations

from typing import Dict, List

from core import Claim, Report, Verdict

DISCLAIMER = (
    "This document is automatically generated research output produced by an "
    "academic prototype. It is not investment advice, not a recommendation to "
    "buy or sell any security, and has not been reviewed by a registered "
    "investment adviser. Every statement above is machine-generated and carries "
    "a machine-assigned confidence score; low-confidence statements have been "
    "withheld rather than shown."
)


def _badge(conf: float) -> str:
    if conf >= 0.85:
        return "HIGH"
    if conf >= 0.70:
        return "MEDIUM"
    return "LOW"


def render_markdown(report: Report, store=None) -> str:
    lines: List[str] = []
    lines.append(f"# Research note — {report.ticker}")
    lines.append("")
    lines.append(f"**Query:** {report.query}  ")
    lines.append(f"**Generated:** {report.generated_at}  ")
    lines.append(f"**Agents invoked:** {', '.join(report.agents_invoked)}  ")
    lines.append(f"**Claim grounding rate:** {report.grounding_rate * 100:.1f}% "
                 f"({report.surfaced_count}/{len(report.verdicts)} claims surfaced)")
    lines.append("")

    for section, claims in report.sections.items():
        if not claims:
            continue
        lines.append(f"## {section}")
        lines.append("")
        for c in claims:
            v = report.verdicts.get(c.claim_id)
            conf = v.confidence if v else 0.0
            srcs = ", ".join(
                (store.get(cid).source if store and store.get(cid) else cid)
                for cid in c.cited_chunk_ids[:2])
            lines.append(f"- {c.text}  ")
            lines.append(f"  <sub>confidence **{conf:.2f}** ({_badge(conf)}) · "
                         f"source: {srcs}</sub>")
        lines.append("")

    flagged = [v for v in report.verdicts.values() if v.status == "flagged"]
    if flagged:
        lines.append("## Withheld by the Verification Agent")
        lines.append("")
        lines.append(f"{len(flagged)} generated claim(s) failed verification and were "
                     f"not surfaced. Reasons recorded:")
        text_of = {c.claim_id: c.text for c, _ in report.withheld}
        for v in flagged[:6]:
            quoted = f"“{text_of[v.claim_id]}” — " if v.claim_id in text_of else ""
            lines.append(f"- {quoted}{v.reason or 'below confidence threshold'} "
                         f"(confidence {v.confidence:.2f})")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"_{DISCLAIMER}_")
    return "\n".join(lines)


def export_pdf(markdown_text: str, path: str) -> bool:
    """Best-effort PDF export. Returns False if no converter is installed."""
    try:
        import subprocess
        subprocess.run(["pandoc", "-f", "markdown", "-o", path, "-"],
                       input=markdown_text.encode("utf-8"), check=True)
        return True
    except Exception:
        return False
