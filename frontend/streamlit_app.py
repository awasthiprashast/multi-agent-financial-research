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


# ----------------------------------------------------------- chatbot vs the data
CHAT_SYSTEM = ("You are a helpful financial assistant. Answer in 6 to 8 sentences. Give "
               "specific current figures (price, moving averages, RSI, volatility, P/E, "
               "margins) wherever you can.")


def chatbot_tab() -> None:
    st.subheader("Ask a chatbot, then check its answer")
    st.markdown(
        "A language model answers freely, the way any chatbot would. Then the Verification "
        "Agent checks every figure and price statement in that answer against live data. "
        "This is what the system exists for: **chatbots sound sure and are often wrong on "
        "numbers.**")
    llm = get_llm_backend()
    if llm is None:
        st.info("No language model is configured on this server, so this tab is switched off. "
                "Add GROQ_API_KEY (free at console.groq.com) in the app's secrets to enable it.")
        return
    ticker = st.selectbox("Company", TICKERS, format_func=lambda t: f"{NAME[t]} ({t})",
                          key="chat_ticker")
    question = st.text_input(
        "Question", f"How is {NAME[ticker]} doing right now, technically and fundamentally? "
                    f"Include specific numbers.", key=f"chat_q_{ticker}")
    if not st.button("Ask and audit", type="primary"):
        return
    try:
        with st.spinner("Asking the chatbot…"):
            answer = llm.generate(CHAT_SYSTEM, question)
        with st.spinner("Checking its answer against live data…"):
            result = audit_text(answer, get_pipeline(), ticker=ticker)
    except (UnknownCompanyError, DataUnavailableError, ValueError) as exc:
        st.error(str(exc))
        return
    except Exception as exc:                          # network or quota at the model provider
        st.error(f"The language model could not be reached ({type(exc).__name__}). Try again.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown(f"**What the chatbot said** ({llm.name})")
        st.write(answer)
    with right:
        st.markdown("**What the data says about it**")
        c = result.counts
        (st.warning if c[CONTRADICTED] + c[UNSUPPORTED] else st.success)(f"**{result.headline}**")
        for s in result.sentences:
            if s.status == UNCHECKABLE:
                continue
            icon, label, _ = BADGE[s.status]
            st.markdown(f"{icon} **{label}.** {s.text}")
            if s.status != VERIFIED and s.explanation:
                st.caption(s.explanation)
    if any(s.correction for s in result.sentences):
        st.markdown("**The same answer, corrected with the real figures:**")
        st.code(result.corrected_text, language=None, wrap_lines=True)
    st.caption("Run it again: a chatbot's answer changes every time, the data does not.")


# ----------------------------------------------------------- accuracy evidence
def evaluation_tab() -> None:
    import json

    st.subheader("How accurate is the verifier?")
    st.markdown("Measured, not claimed. Every number below is produced by scripts in `eval/` "
                "and stored in `results/`. Limits are stated next to each result.")
    res = Path(__file__).resolve().parents[1] / "results"

    def load(name):
        try:
            return json.loads((res / name).read_text(encoding="utf-8"))
        except Exception:
            return None

    metrics, mistral, groq, router = (load("metrics.json"), load("llm_eval_summary.json"),
                                      load("groq_llm_eval_summary.json"),
                                      load("router_heldout.json"))
    if metrics:
        h = metrics["hallucination"]
        st.markdown("#### 1. Can it catch planted errors? (162 claims, synthetic company data)")
        a, b, c, d = st.columns(4)
        a.metric("Errors caught", f"{h['catch_rate_recall'] * 100:.0f}%")
        b.metric("Correct claims wrongly rejected", f"{h['false_positive_rate'] * 100:.0f}%")
        c.metric("Time per claim", f"{h['ms_per_claim']:.1f} ms")
        d.metric("Claims tested", h["benchmark_size"])
        st.caption("The planted errors are the kinds the verifier was built for (changed "
                   "numbers, wrong company, invented details), so this shows the mechanism "
                   "works, not that it is perfect on all real text. The next test is the one "
                   "that matters.")
        rows = {k: f"{v['catch_rate'] * 100:.0f}%" for k, v in metrics["ablation"].items()}
        st.markdown("**What each part contributes** (share of errors caught when a part is removed):")
        st.table({"Configuration": list(rows), "Errors caught": list(rows.values())})
    if mistral and groq:
        st.markdown("#### 2. Errors nobody planted: claims written by real language models")
        st.table({
            "": ["Claims written", "Rejected on first pass", "Rate", "Wrong numbers that reached the report"],
            "Mistral 7B (small, local)": [str(mistral["llm_claims"]), str(mistral["claims_flagged_first_pass"]),
                                          f"{mistral['first_pass_flag_rate'] * 100:.1f}%",
                                          str(mistral["wrong_numbers_that_reached_the_report"])],
            "gpt-oss-120b (large, Groq)": [str(groq["llm_claims"]), str(groq["claims_flagged_first_pass"]),
                                           f"{groq['first_pass_flag_rate'] * 100:.1f}%",
                                           str(groq["wrong_numbers_that_reached_the_report"])],
        })
        st.caption("The small model often got a price direction wrong (for example "
                   "\"below the 20-day average\" when it was above); the verifier caught those "
                   "using the raw price series. The large model rarely erred, which shows the "
                   "verifier does not hold back correct claims. Three companies, five runs "
                   "each, synthetic evidence. A hand audit of the output is in "
                   "`results/llm_manual_audit.md`.")
    if router:
        st.markdown("#### 3. Does the router pick the right specialists?")
        a, b, c = st.columns(3)
        a.metric("Exact match, 30 unseen queries", f"{router['exact_match'] * 100:.0f}%")
        b.metric("Specialists recalled", f"{router['label_recall'] * 100:.0f}%")
        c.metric("Agent calls saved", f"{router['agent_invocation_reduction_pct']:.0f}%")
    st.markdown("#### What it cannot do")
    st.markdown("- Judge **conclusions**: \"above its average, so the stock will fall\" has a true "
                "premise and a wrong conclusion; the verifier checks the premise.\n"
                "- Check **forecasts, opinions** or text with no figure in it. These are "
                "reported as not checkable, never as correct.\n"
                "- Cover companies outside the 12 listed, or claims about things the data "
                "sources do not hold.")


# ------------------------------------------------------------------------- page
st.title("Multi-Agent Financial Research Assistant")

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

tab_brief, tab_check, tab_chat, tab_eval = st.tabs(
    ["Company briefing", "Fact-check a text", "Chatbot vs the data", "How accurate is it?"])
with tab_brief:
    briefing_tab()
with tab_check:
    audit_tab()
with tab_chat:
    chatbot_tab()
with tab_eval:
    evaluation_tab()
