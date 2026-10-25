"""
frontend/streamlit_app.py — the demo UI.  Run: streamlit run frontend/streamlit_app.py

Two ways in:
  * Audit a text — paste anything written about a listed company (a broker note, a
    chatbot answer) and see which sentences the data supports, which it contradicts
    and which cannot be checked at all.
  * Research report — the multi-agent pipeline writes a report and verifies it.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from agents.auditor import (  # noqa: E402
    CONTRADICTED,
    UNCHECKABLE,
    UNSUPPORTED,
    VERIFIED,
    audit_text,
)
from data.corpus import DataUnavailableError  # noqa: E402
from data.universe import NAME, UnknownCompanyError, supported_tickers  # noqa: E402
from pipeline import ResearchPipeline  # noqa: E402

OFFLINE = os.environ.get("FINRESEARCH_OFFLINE", "") in ("1", "true", "yes")
TICKERS = supported_tickers(offline=OFFLINE)

st.set_page_config(page_title="Claim Auditor for Financial Text", layout="wide")


@st.cache_resource
def get_pipeline(threshold: float, analysts: str = "template"):
    return ResearchPipeline(threshold=threshold, analysts=analysts, prefer_live=not OFFLINE)


@st.cache_resource
def llm_available() -> bool:
    from llm import get_llm
    return get_llm().available


def build_example(pipe) -> str:
    """A fresh example from today's numbers for a random company: some sentences are
    true, some carry a changed number or a wrong direction, one is a forecast."""
    ticker = random.choice(TICKERS)
    ind = pipe._prepare(ticker)["indicators"]
    name = NAME[ticker]
    above50 = ind["last_close"] > ind["sma_50"]
    above20 = ind["last_close"] > ind["sma_20"]
    return " ".join([
        f"{name} closed {'above' if above50 else 'below'} its 50-day average.",
        f"The RSI is {ind['rsi_14']}.",
        f"{name} annualised volatility of daily returns is {ind['annual_vol_pct'] * 1.6:.1f}%.",
        f"{name} closed {'below' if above20 else 'above'} its 20-day average.",
        f"{name} shares will double within two years.",
    ])


EXAMPLE_FALLBACK = (
    "TCS closed above its 50-day average. The RSI is 91. "
    "Annualised volatility of daily returns is 5.0%. TCS shares will double within two years."
)

BADGE = {
    VERIFIED: ("✅", "Verified", "success"),
    CONTRADICTED: ("❌", "Contradicted", "error"),
    UNSUPPORTED: ("⚠️", "Unsupported", "warning"),
    UNCHECKABLE: ("➖", "Not checkable", "info"),
}


def show_evidence(rows) -> None:
    for e in rows:
        st.markdown(f"**{e.source}** · {e.doc_type} · {e.published}")
        st.write(e.text)


def audit_tab(threshold: float) -> None:
    st.subheader("Audit a text against real data")
    st.markdown(
        "Paste a paragraph about one of the listed companies. Every sentence is checked "
        "against Yahoo Finance prices and statements and recent news. Numbers must match "
        "the data (same value, same label), and statements like *“closed above its "
        "50-day average”* are recomputed from the raw price series. Forecasts and opinions "
        "are **never** marked as verified.")

    if st.button("Load an example"):
        try:
            with st.spinner("Building an example from today's data…"):
                st.session_state["audit_text"] = build_example(get_pipeline(threshold))
        except DataUnavailableError:
            st.session_state["audit_text"] = EXAMPLE_FALLBACK
    text = st.text_area("Text to audit", key="audit_text", height=180, max_chars=8000,
                        placeholder="Paste an analyst note, news summary or chatbot answer…")
    ticker_choice = st.selectbox(
        "Company the text is about", ["Detect from the text"] + TICKERS,
        format_func=lambda t: t if t.startswith("Detect") else f"{NAME[t]} ({t})")

    if not st.button("Audit", type="primary"):
        return
    if not text.strip():
        st.warning("Paste some text first.")
        return
    try:
        with st.spinner("Fetching data and checking each sentence…"):
            result = audit_text(
                text, get_pipeline(threshold),
                ticker=None if ticker_choice.startswith("Detect") else ticker_choice)
    except (UnknownCompanyError, DataUnavailableError, ValueError) as exc:
        st.error(str(exc))
        return

    counts = result.counts
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Verified", counts[VERIFIED])
    c2.metric("Contradicted", counts[CONTRADICTED])
    c3.metric("Unsupported", counts[UNSUPPORTED])
    c4.metric("Not checkable", counts[UNCHECKABLE])
    c5.metric("Accuracy of checkable", "n/a" if result.accuracy is None
              else f"{result.accuracy * 100:.0f}%")
    st.caption(f"{result.checkable} of {len(result.sentences)} sentences could be checked "
               f"({result.coverage * 100:.0f}% coverage) · companies: "
               + ", ".join(NAME[t] for t in result.tickers)
               + (" · text truncated" if result.truncated else ""))

    for s in result.sentences:
        icon, label, kind = BADGE[s.status]
        getattr(st, kind)(f"{icon} **{label}** — {s.text}")
        notes = []
        for n in s.numbers:
            if n.matched:
                notes.append(f"`{n.raw}` matches the data")
            elif n.actual:
                notes.append(f"`{n.raw}` is not in the data; closest value for the same "
                             f"quantity is **{n.actual}**")
            else:
                notes.append(f"`{n.raw}` is not in the data")
        notes += [f"direction confirmed: {d}" for d in s.directions_confirmed]
        notes += [f"direction contradicted: {d}" for d in s.directions_contradicted]
        if s.reason and not s.numbers:
            notes.append(s.reason)
        elif s.reason and s.status == UNCHECKABLE:
            notes.append(s.reason)
        if notes:
            st.caption(" · ".join(notes))
        if s.evidence and s.status != UNCHECKABLE:
            with st.expander("Evidence this was checked against"):
                show_evidence(s.evidence)

    st.info("“Verified” means the numbers and indicator statements match the data. It does "
            "not mean the sentence's conclusion is right. Research prototype, not "
            "investment advice.")


def report_tab(threshold: float) -> None:
    st.subheader("Multi-agent research report")
    ticker = st.selectbox("Company", TICKERS, format_func=lambda t: f"{NAME[t]} ({t})",
                          key="report_ticker")
    has_llm = llm_available()
    use_llm = st.checkbox("LLM analysts", value=False, disabled=not has_llm,
                          help="Analysts write claims with a language model; every claim is "
                               "still verified. Needs GROQ_API_KEY (or a local Ollama).")
    if not has_llm:
        st.caption("LLM analysts are off: no language-model backend is configured on this "
                   "server. The deterministic template analysts are used.")
    query = st.text_input("Research question", f"Give me a full report on {NAME[ticker]}")

    if st.button("Run research", type="primary"):
        st.session_state["ran"] = True

    # Results live behind session state: Streamlit re-runs the script on every widget
    # interaction, so gating them on the button alone would make them vanish.
    if not st.session_state.get("ran"):
        return
    try:
        pipe = get_pipeline(threshold, "llm" if use_llm else "template")
        with st.spinner("Agents working…"):
            rep = pipe.run(query, ticker=ticker)
    except (RuntimeError, UnknownCompanyError, DataUnavailableError) as exc:
        st.error(str(exc))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grounding rate", f"{rep.grounding_rate * 100:.0f}%")
    c2.metric("Claims surfaced", f"{rep.surfaced_count}/{len(rep.verdicts)}")
    c3.metric("Agents invoked", len(rep.agents_invoked))
    c4.metric("Latency", f"{rep.timings['total_s'] * 1000:.0f} ms")
    st.caption(f"Router selected: {', '.join(rep.agents_invoked)} · "
               f"corpus: {rep.mode} · analysts: {rep.analysts}")

    store = rep.store
    for section, claims in rep.sections.items():
        st.markdown(f"#### {section}")
        for c in claims:
            v = rep.verdicts[c.claim_id]
            bar = "🟢" if v.confidence >= 0.85 else "🟡" if v.confidence >= 0.7 else "🟠"
            st.markdown(f"{bar} {c.text}  \n"
                        f"<span style='color:#666;font-size:0.85em'>confidence "
                        f"{v.confidence:.2f} · numeric support {v.numeric_support:.2f} "
                        f"· semantic {v.semantic_support:.2f}</span>",
                        unsafe_allow_html=True)
            with st.expander("Show the evidence this was checked against"):
                for cid in v.supporting_chunk_ids:
                    ch = store.get(cid) if store else None
                    if ch:
                        st.markdown(f"**{ch.source}** · {ch.doc_type} · {ch.published}")
                        st.write(ch.text)

    st.markdown("#### Withheld by the Verification Agent")
    if rep.withheld:
        for claim, v in rep.withheld:
            st.error(f"“{claim.text}”\n\nconfidence {v.confidence:.2f} — {v.reason}")
    else:
        st.success("Nothing was withheld: every generated claim passed verification.")
    st.warning("Research output from an academic prototype. Not investment advice.")


st.title("Claim Auditor for Financial Text")
st.caption("Paste what someone (or a chatbot) wrote about a stock and see which sentences "
           "the data actually supports. Research prototype · not investment advice.")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence threshold θ", 0.0, 1.0, 0.60, 0.05,
                          help="Used by the research-report tab; claims scoring below it "
                               "are withheld.")
    st.divider()
    st.markdown("**How a sentence is checked**\n\n"
                "1. numbers matched to data with their label\n"
                "2. indicators recomputed from raw prices\n"
                "3. direction statements tested\n"
                "4. company named must match the data\n"
                "5. forecasts and opinions set aside")
    st.caption(f"Data: {'offline snapshot' if OFFLINE else 'live Yahoo Finance + Google News'} "
               f"· {len(TICKERS)} companies")

tab_audit, tab_report = st.tabs(["Audit a text", "Research report"])
with tab_audit:
    audit_tab(threshold)
with tab_report:
    report_tab(threshold)
