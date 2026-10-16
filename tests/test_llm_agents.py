"""LLM analyst agents, exercised with a scripted fake model (no network, no GPU)."""

import json

from agents.llm_analysts import LLMAnalyst, _parse_json
from llm import LLM
from pipeline import ResearchPipeline

PIPE = ResearchPipeline(prefer_live=False)


class ScriptedLLM(LLM):
    """Returns canned JSON; records every prompt it was sent."""
    name = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    @property
    def available(self):
        return True

    def generate(self, system, user, json_mode=False):
        self.prompts.append(user)
        return self.replies.pop(0) if self.replies else "{}"


def _agent(kind, *replies):
    ctx = PIPE._prepare("TCS.NS")
    llm = ScriptedLLM(*replies)
    return LLMAnalyst(ctx["store"], kind, llm), llm, ctx


def test_json_parser_tolerates_chatter_around_the_object():
    assert _parse_json('Sure! {"claims": []} hope that helps') == {"claims": []}
    assert _parse_json("not json") is None


def test_llm_claims_carry_the_citations_the_model_gave():
    reply = json.dumps({"claims": [{"text": "TCS reported return on equity of 51.2% for the year.",
                                    "sources": [1]}]})
    agent, llm, ctx = _agent("fundamental", reply)
    claims = agent.run("TCS.NS", ctx)
    assert len(claims) == 1 and claims[0].cited_chunk_ids
    assert "Evidence:" in llm.prompts[0]            # the model was shown retrieved evidence


def test_llm_hallucinated_number_is_withheld_after_failed_revision():
    bad = {"claims": [{"text": "TCS reported return on equity of 74.9% for the year.",
                       "sources": [1]}]}
    again = {"text": "TCS reported return on equity of 74.9% for the year.", "sources": [1]}
    agent, llm, ctx = _agent("fundamental", json.dumps(bad), json.dumps(again), json.dumps(again))
    pipe_claims = agent.run("TCS.NS", ctx)
    from agents.validator import VerificationAgent
    v = VerificationAgent(ctx["store"], max_retries=2)
    surfaced, verdicts, stats = v.verify_with_revision(
        pipe_claims, {"fundamental": agent}, "TCS.NS", ctx)
    assert surfaced == [] and stats["dropped"] == 1
    assert stats["revisions_attempted"] == 2          # bounded
    assert v.withheld and "74.9" in v.withheld[0][0].text


def test_llm_claim_is_repaired_when_the_model_corrects_itself():
    bad = {"claims": [{"text": "TCS reported return on equity of 74.9% for the year.",
                       "sources": [1]}]}
    fixed = {"text": "TCS reported return on equity of 51.2% for the year.", "sources": [1]}
    agent, llm, ctx = _agent("fundamental", json.dumps(bad), json.dumps(fixed))
    from agents.validator import VerificationAgent
    surfaced, verdicts, stats = VerificationAgent(ctx["store"]).verify_with_revision(
        agent.run("TCS.NS", ctx), {"fundamental": agent}, "TCS.NS", ctx)
    assert len(surfaced) == 1 and "51.2" in surfaced[0].text
    assert "74.9" in llm.prompts[1]                   # the rejected claim was shown back
    assert verdicts[surfaced[0].claim_id].status == "revised"


def test_contradicted_direction_in_llm_claim_is_withheld():
    ind = PIPE._prepare("TCS.NS")["indicators"]
    wrong = "above" if ind["last_close"] < ind["sma_20"] else "below"
    reply = json.dumps({"claims": [{"text": f"The closing price of TCS.NS is {wrong} the "
                                            "20-day simple moving average.", "sources": [1]}]})
    agent, _, ctx = _agent("technical", reply)
    from agents.validator import VerificationAgent
    claims = agent.run("TCS.NS", ctx)
    assert VerificationAgent(ctx["store"]).verify(claims[0]).status == "flagged"


def test_pipeline_runs_end_to_end_with_llm_analysts():
    def reply(ticker):
        return json.dumps({"claims": [{"text": "The annualised volatility of TCS.NS daily "
                                               "returns over the window was "
                                               f"{PIPE._prepare(ticker)['indicators']['annual_vol_pct']}%.",
                                       "sources": [1]}]})
    pipe = ResearchPipeline(prefer_live=False, analysts="llm", llm=ScriptedLLM(reply("TCS.NS")))
    rep = pipe.run("How volatile is TCS?", ticker="TCS.NS", force_agents=["risk"])
    assert rep.analysts == "llm" and rep.surfaced_count == 1


def test_llm_mode_without_a_backend_fails_loudly():
    import pytest
    with pytest.raises(RuntimeError):
        ResearchPipeline(prefer_live=False, analysts="llm", llm=LLM())
