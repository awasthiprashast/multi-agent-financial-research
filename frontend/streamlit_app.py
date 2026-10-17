"""
frontend/streamlit_app.py — the demo UI.  Run: streamlit run frontend/streamlit_app.py

Built around one idea: the interesting part of this system is not the report, it
is the audit trail. So the UI shows the confidence on every sentence, lets the
reviewer expand the exact source chunk behind it, and — the slide that lands —
shows what the Verification Agent refused to print.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from data.corpus import DataUnavailableError  # noqa: E402
from data.universe import COMPANIES, NAME, UnknownCompanyError  # noqa: E402
from pipeline import ResearchPipeline  # noqa: E402

st.set_page_config(page_title="Multi-Agent Financial Research Assistant",
                   layout="wide")


@st.cache_resource
def get_pipeline(threshold: float, analysts: str = "template"):
    return ResearchPipeline(threshold=threshold, analysts=analysts)


st.title("Multi-Agent Financial Research Assistant")
st.caption("Every claim is re-checked against its source before you see it. "
           "BCSE497J Project-I · SCOPE, VIT")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Confidence threshold θ", 0.0, 1.0, 0.60, 0.05,
                          help="Claims scoring below this are withheld.")
    ticker = st.selectbox("Company", [c.ticker for c in COMPANIES],
                          format_func=lambda t: f"{NAME[t]} ({t})",
                          help="Live data. Only TCS, Reliance and Infosys have an offline "
                               "synthetic snapshot.")
    use_llm = st.checkbox("LLM analysts", value=False,
                          help="Analysts write claims with a language model (Groq if "
                               "GROQ_API_KEY is set, else local Ollama). Slower; every "
                               "claim is still verified.")
    st.divider()
    st.markdown("**Verification mechanism**")
    st.markdown("1. independent re-retrieval\n2. numeric adjudication\n"
                "3. semantic adjudication\n4. citation consistency\n"
                "5. entity binding\n6. bounded revision loop")

query = st.text_input("Research question",
                      "Should I look at TCS for a 1-year hold?")

if st.button("Run research", type="primary"):
    st.session_state["ran"] = True

# The results live behind session state rather than behind the button itself.
# Streamlit re-runs the whole script on every widget interaction, so gating the
# results on `if st.button(...)` would make them disappear the moment anyone
# typed in the probe box below or moved the threshold slider.
if st.session_state.get("ran"):
    try:
        pipe = get_pipeline(threshold, "llm" if use_llm else "template")
    except RuntimeError as exc:
        st.error(f"{exc}")
        st.stop()
    try:
        with st.spinner("Agents working…"):
            rep = pipe.run(query, ticker=ticker)
    except (UnknownCompanyError, DataUnavailableError) as exc:
        st.error(str(exc))
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grounding rate", f"{rep.grounding_rate * 100:.0f}%")
    c2.metric("Claims surfaced",
              f"{rep.surfaced_count}"
              f"/{len(rep.verdicts)}")
    c3.metric("Agents invoked", len(rep.agents_invoked))
    c4.metric("Latency", f"{rep.timings['total_s'] * 1000:.0f} ms")

    st.caption(f"Router selected: {', '.join(rep.agents_invoked)} · "
               f"corpus: {rep.mode} · analysts: {rep.analysts}")

    store = rep.store
    for section, claims in rep.sections.items():
        st.subheader(section)
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

    st.subheader("Withheld by the Verification Agent")
    if rep.withheld:
        for claim, v in rep.withheld:
            st.error(f"“{claim.text}”\n\nconfidence {v.confidence:.2f} — {v.reason}")
    else:
        st.success("Nothing was withheld: every generated claim passed verification.")

    st.divider()
    st.warning("Research output from an academic prototype. Not investment advice.")


# --------------------------------------------------------------------------
# The probe sits outside the results block so it is available at any time and
# survives a re-run. This is the demonstration that lands: paste a true claim,
# then change one digit.
# --------------------------------------------------------------------------
st.divider()
st.subheader("Check any claim yourself")
st.caption("Paste a statement about the selected company. Change a single digit "
           "and watch the confidence collapse — the sentence still reads perfectly.")

probe = st.text_input("Claim to verify", key="probe",
                      placeholder="Operating margin for the year was 24.3%, a movement "
                                  "of -70 bps against the prior year.")

if probe:
    from agents.validator import VerificationAgent
    from core import Claim

    s = get_pipeline(threshold, "template")._prepare(ticker)["store"]
    cl = Claim.new("external", "ad-hoc", probe)
    cl.cited_chunk_ids = [c.chunk_id for c, _ in s.query(probe, k=3, ticker=ticker)]
    verdict = VerificationAgent(s, threshold=threshold).verify(cl)

    passed = verdict.status == "verified"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Verdict", "ACCEPTED" if passed else "WITHHELD")
    m2.metric("Confidence", f"{verdict.confidence:.2f}")
    m3.metric("Numeric support", f"{verdict.numeric_support:.2f}")
    m4.metric("Semantic support", f"{verdict.semantic_support:.2f}")

    if passed:
        st.success("Every quantity in this claim was matched against retrieved evidence.")
    else:
        st.error(verdict.reason or "Below the confidence threshold.")

    with st.expander("Evidence the claim was checked against"):
        for cid in verdict.supporting_chunk_ids:
            ch = s.get(cid)
            if ch:
                st.markdown(f"**{ch.source}** · {ch.doc_type} · {ch.published}")
                st.write(ch.text)
