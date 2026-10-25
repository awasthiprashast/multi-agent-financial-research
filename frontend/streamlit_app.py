"""
frontend/streamlit_app.py — the demo UI.  Run: streamlit run frontend/streamlit_app.py

Two tabs:
  * Company briefing — a plain-English summary of one company, built from verified market
    data. If a language model is configured it writes the summary; every sentence it writes
    is audited against the data and dropped if unsupported.
  * Check a text — paste anything written about a listed company (a broker note, a chatbot
    answer) and see which sentences the data supports, which it contradicts, and the text
    corrected with the real figures.
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
from agents.briefing import make_briefing  # noqa: E402
from data.corpus import DataUnavailableError  # noqa: E402
from data.universe import NAME, UnknownCompanyError, supported_tickers  # noqa: E402
from pipeline import ResearchPipeline  # noqa: E402

OFFLINE = os.environ.get("FINRESEARCH_OFFLINE", "") in ("1", "true", "yes")
TICKERS = supported_tickers(offline=OFFLINE)
THRESHOLD = 0.60

st.set_page_config(page_title="Multi-Agent Financial Research Assistant", layout="wide")


@st.cache_resource
def get_pipeline():
    return ResearchPipeline(threshold=THRESHOLD, analysts="template", prefer_live=not OFFLINE)


@st.cache_resource
def get_llm_backend():
    from llm import get_llm
    llm = get_llm()
    return llm if llm.available else None


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

GLOSSARY = {
    "Moving average (20/50/200-day)": "The average closing price over that many trading days. "
        "A price above its average usually means the recent trend is up.",
    "RSI": "Relative Strength Index, 0 to 100. Above 70 is called overbought (rose fast), "
        "below 30 oversold (fell fast), in between neutral.",
    "MACD": "Compares a short and a long moving average. The MACD line above its signal line "
        "is read as upward momentum.",
    "Volatility": "How much the price swings, as a yearly percentage. Higher means a bumpier ride.",
    "Maximum drawdown": "The biggest fall from a peak to a later low in the last year.",
    "P/E": "Price divided by yearly earnings per share: roughly how many years of profit the "
        "price pays for.",
    "ROE": "Return on equity: profit as a share of shareholders' money.",
    "Debt-to-equity": "Borrowings relative to shareholders' money. Lower means less leverage.",
}

BADGE = {
    VERIFIED: ("✅", "Correct", "success"),
    CONTRADICTED: ("❌", "Wrong", "error"),
    UNSUPPORTED: ("⚠️", "Not confirmed", "warning"),
    UNCHECKABLE: ("➖", "Can't be checked", "info"),
}


def show_evidence(rows) -> None:
    for e in rows:
        st.markdown(f"**{e.source}** · {e.doc_type} · {e.published}")
        st.write(e.text)


# ------------------------------------------------------------------ check a text
def audit_tab() -> None:
    st.subheader("Fact-check any text about a company")
    st.markdown(
        "Paste a paragraph from an analyst note, a news site or a chatbot. Each sentence is "
        "checked against live prices, company statements and recent news. You get what is "
        "right, what is wrong, and a corrected version.")

    if st.button("Try an example"):
        try:
            with st.spinner("Building an example from today's data…"):
                st.session_state["audit_text"] = build_example(get_pipeline())
        except DataUnavailableError:
            st.session_state["audit_text"] = EXAMPLE_FALLBACK
    text = st.text_area("Text to check", key="audit_text", height=170, max_chars=8000,
                        placeholder="Paste something written about a company, e.g. “Infosys "
                                    "closed above its 50-day average and its RSI is 72.”")
    ticker_choice = st.selectbox(
        "Which company is it about?", ["Work it out from the text"] + TICKERS,
        format_func=lambda t: t if t.startswith("Work") else f"{NAME[t]} ({t})")

    if not st.button("Check it", type="primary"):
        return
    if not text.strip():
        st.warning("Paste some text first.")
        return
    try:
        with st.spinner("Fetching data and checking each sentence…"):
            result = audit_text(
                text, get_pipeline(),
                ticker=None if ticker_choice.startswith("Work") else ticker_choice)
    except (UnknownCompanyError, DataUnavailableError, ValueError) as exc:
        st.error(str(exc))
        return

    c = result.counts
    wrong = c[CONTRADICTED] + c[UNSUPPORTED]
    banner = st.success if result.checkable and not wrong else st.warning if wrong else st.info
    banner(f"**{result.headline}**")
    st.caption(f"✅ {c[VERIFIED]} correct · ❌ {c[CONTRADICTED]} wrong · "
               f"⚠️ {c[UNSUPPORTED]} not confirmed · ➖ {c[UNCHECKABLE]} can't be checked "
               f"(opinions, forecasts, or no figure to test) · "
               + ", ".join(NAME[t] for t in result.tickers)
               + (" · text was shortened" if result.truncated else ""))

    if any(s.correction for s in result.sentences):
        st.markdown("**Corrected version** (wrong figures and directions replaced with what "
                    "the data says):")
        st.code(result.corrected_text, language=None, wrap_lines=True)

    st.markdown("**Sentence by sentence**")
    for s in result.sentences:
        icon, label, kind = BADGE[s.status]
        getattr(st, kind)(f"{icon} **{label}.** {s.text}")
        if s.explanation:
            st.caption(s.explanation)
        if s.correction:
            st.caption(f"Should read: *{s.correction}*")
        if s.evidence and s.status != UNCHECKABLE:
            with st.expander("Data it was checked against"):
                show_evidence(s.evidence)
    st.caption("“Correct” means the figures and price statements match the data. It does not "
               "mean the writer's conclusion is right. Not investment advice.")


# ---------------------------------------------------------------- company briefing
def key_numbers(ticker: str, ctx: dict) -> list:
    ind = ctx["indicators"]
    rows = [("Last price", f"₹{ind['last_close']:,.2f}"),
            ("1-year return", f"{ind['period_return_pct']:+.1f}%"),
            ("Volatility (yearly)", f"{ind['annual_vol_pct']:.1f}%"),
            ("Worst fall (1 yr)", f"{ind['max_drawdown_pct']:.1f}%"),
            ("RSI (14)", f"{ind['rsi_14']}")]
    try:
        from data.connectors.live import fetch_fundamentals
        f = fetch_fundamentals(ticker) if ctx["mode"] == "live" else None
    except Exception:
        f = None
    if f:
        if f.get("pe"):
            rows.append(("P/E", f"{f['pe']:.1f}"))
        if f.get("roe"):
            rows.append(("Return on equity", f"{f['roe']:.1f}%"))
        if f.get("profit_margin"):
            rows.append(("Profit margin", f"{f['profit_margin']:.1f}%"))
    return rows


def briefing_tab() -> None:
    st.subheader("What does the data say about a company?")
    ticker = st.selectbox("Company", TICKERS, format_func=lambda t: f"{NAME[t]} ({t})",
                          key="brief_ticker")
    question = st.text_input("Your question", f"What does the data say about {NAME[ticker]} right now?")

    if st.button("Get briefing", type="primary"):
        pipe = get_pipeline()
        try:
            with st.spinner("Collecting prices, financials and news…"):
                rep = pipe.run(question, ticker=ticker)
            with st.spinner("Writing and checking the summary…"):
                brief = make_briefing(pipe, rep, question, get_llm_backend())
            st.session_state["brief"] = (ticker, rep, brief, key_numbers(ticker, pipe._prepare(ticker)))
        except (UnknownCompanyError, DataUnavailableError, RuntimeError) as exc:
            st.session_state.pop("brief", None)
            st.error(str(exc))
            return

    saved = st.session_state.get("brief")
    if not saved or saved[0] != ticker:
        return
    _, rep, brief, numbers = saved

    cols = st.columns(len(numbers))
    for col, (label, value) in zip(cols, numbers):
        col.metric(label, value)

    st.markdown("#### Summary")
    for s in brief.sentences:
        st.markdown(f"- {s.text}")
    if brief.source.startswith("llm:"):
        st.caption(f"Written by {brief.source[4:]} from verified data only. Every sentence was "
                   f"checked against the data first; {brief.dropped} "
                   f"{'sentence was' if brief.dropped == 1 else 'sentences were'} removed "
                   f"because the data could not confirm {'it' if brief.dropped == 1 else 'them'}.")
    else:
        st.caption("Built directly from verified data (no language model is configured on this "
                   "server). " + " ".join(brief.notes))

    with st.expander("Full fact sheet (every figure verified)"):
        for section, claims in rep.sections.items():
            st.markdown(f"**{section}**")
            for cl in claims:
                v = rep.verdicts[cl.claim_id]
                st.markdown(f"- {cl.text}  \n  <span style='color:#777;font-size:0.85em'>"
                            f"verified, confidence {v.confidence:.2f}</span>",
                            unsafe_allow_html=True)
        if rep.withheld:
            st.markdown("**Withheld because the data did not support them**")
            for cl, v in rep.withheld:
                st.error(f"“{cl.text}” — {v.reason}")
        st.caption(f"Data: {rep.mode} · agents used: {', '.join(rep.agents_invoked)} · "
                   f"{rep.timings['total_s']:.1f}s")

    with st.expander("What do these terms mean?"):
        for term, meaning in GLOSSARY.items():
            st.markdown(f"**{term}.** {meaning}")
    st.warning("Information from market data, not investment advice.")


# ------------------------------------------------------------------------- page
st.title("Multi-Agent Financial Research Assistant")
st.caption("Plain-English company briefings, and a fact-checker for anything written about a "
           "listed company. Every number is verified against live market data before you see "
           "it.")

with st.sidebar:
    st.header("How it works")
    st.markdown("Specialist agents (fundamental, technical, sentiment, risk) gather facts. A "
                "separate **Verification Agent** re-checks each one against the source data "
                "and withholds anything it cannot support. The same verifier checks text "
                "written by others.")
    st.markdown("**A sentence is checked by:**\n\n"
                "1. matching its numbers to the data\n"
                "2. recomputing indicators from raw prices\n"
                "3. testing up/down statements\n"
                "4. making sure it's about the right company\n"
                "5. setting aside forecasts and opinions")
    llm = get_llm_backend()
    st.caption(f"Data: {'offline sample' if OFFLINE else 'live Yahoo Finance + Google News'} · "
               f"{len(TICKERS)} companies · summaries: "
               + (f"written by {llm.name}" if llm else "rule-based"))

tab_brief, tab_check = st.tabs(["Company briefing", "Fact-check a text"])
with tab_brief:
    briefing_tab()
with tab_check:
    audit_tab()
