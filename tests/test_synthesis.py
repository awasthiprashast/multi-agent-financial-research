"""Overall view: deterministic synthesis whose every sentence is independently verified."""

from agents.synthesis import SynthesisAgent, _tone, _trend
from agents.validator import VerificationAgent
from core import Claim
from pipeline import ResearchPipeline

PIPE = ResearchPipeline(prefer_live=False)          # synthesis on (the default)
ALL4 = ["fundamental", "technical", "sentiment", "risk"]


def test_full_report_leads_with_a_verified_overall_view():
    rep = PIPE.run("Complete analysis of Infosys")
    assert list(rep.sections)[0] == "Overall view"
    assert rep.grounding_rate == 1.0
    text = " ".join(c.text for c in rep.sections["Overall view"])
    assert "price action" in text and "net sentiment score" in text and "annualised volatility" in text


def test_overall_view_needs_at_least_two_domains():
    assert "Overall view" not in PIPE.run("How volatile is Reliance?").sections          # one agent
    assert "Overall view" in PIPE.run("Complete analysis of TCS").sections


def test_synthesis_can_be_switched_off():
    off = ResearchPipeline(prefer_live=False, synthesize=False)
    assert "Overall view" not in off.run("Complete analysis of TCS").sections


def test_every_synthesis_sentence_is_grounded_in_cited_evidence():
    ctx = PIPE._prepare("TCS.NS")
    claims = SynthesisAgent(ctx["store"], set(ALL4)).run("TCS.NS", ctx)
    assert len(claims) == 3                       # snapshot has no financials: no growth-vs-price claim
    v = VerificationAgent(ctx["store"])
    for c in claims:
        assert c.cited_chunk_ids, c.text
        assert v.verify(c).status == "verified", c.text


def test_a_wrong_premise_in_the_synthesis_is_caught():
    """Tamper with each kind of fact the synthesis states; the verifier rejects it."""
    ctx = PIPE._prepare("TCS.NS")
    store, ind = ctx["store"], ctx["indicators"]
    claims = SynthesisAgent(store, set(ALL4)).run("TCS.NS", ctx)
    technical, sentiment, risk = claims
    v = VerificationAgent(store)

    flipped = technical.text.replace("below both", "above both") if "below both" in technical.text \
        else technical.text.replace("above both", "below both")
    assert flipped != technical.text
    assert v.verify(Claim.new("t", "s", flipped, technical.cited_chunk_ids)).status == "flagged"

    wrong_vol = risk.text.replace(f"{ind['annual_vol_pct']}%", f"{ind['annual_vol_pct'] + 9}%")
    assert v.verify(Claim.new("t", "s", wrong_vol, risk.cited_chunk_ids)).status == "flagged"

    wrong_net = sentiment.text.replace("net sentiment score is 0.0", "net sentiment score is 0.8")
    assert wrong_net != sentiment.text
    assert v.verify(Claim.new("t", "s", wrong_net, sentiment.cited_chunk_ids)).status == "flagged"


def test_agree_or_diverge_wording_follows_the_facts():
    assert _tone(0.33) == "positive" and _tone(-0.5) == "negative" and _tone(0.05) == "neutral"
    base = {"sma_20": 100.0, "sma_50": 100.0}
    assert _trend({**base, "last_close": 110.0}) == "above"
    assert _trend({**base, "last_close": 90.0}) == "below"
    assert _trend({"sma_20": 100.0, "sma_50": 120.0, "last_close": 110.0}) == "mixed"


def test_neutral_news_never_claims_agreement():
    ctx = PIPE._prepare("TCS.NS")                      # snapshot sentiment is net 0.0
    sent = [c for c in SynthesisAgent(ctx["store"], set(ALL4)).run("TCS.NS", ctx)
            if "News tone" in c.text][0]
    assert "neutral" in sent.text and "do not point clearly" in sent.text


def test_synthesis_gives_no_advice():
    from agents.llm_analysts import is_publishable
    for t in ("TCS.NS", "RELIANCE.NS", "INFY.NS"):
        ctx = PIPE._prepare(t)
        for c in SynthesisAgent(ctx["store"], set(ALL4)).run(t, ctx):
            assert is_publishable(c.text), c.text
