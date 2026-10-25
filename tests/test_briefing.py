"""Plain-English briefing: audited before it is shown."""

import pytest

from agents.auditor import audit_text, normalise
from agents.briefing import make_briefing
from llm import LLM
from pipeline import ResearchPipeline


class FakeLLM(LLM):
    name = "fake/test"

    def __init__(self, reply):
        self.reply = reply

    @property
    def available(self):
        return True

    def generate(self, system, user, json_mode=False):
        return self.reply


@pytest.fixture(scope="module")
def pipe():
    return ResearchPipeline(prefer_live=False, analysts="template")


@pytest.fixture(scope="module")
def report(pipe):
    return pipe.run("What does the data say about TCS?", ticker="TCS.NS")


def test_unsupported_sentences_written_by_the_model_are_dropped(pipe, report):
    reply = ("TCS reported an operating margin of 24.3% for the year. "
             "TCS reported an operating margin of 31.9% for the year. "
             "TCS could be risky if earnings falter.")
    b = make_briefing(pipe, report, "q", FakeLLM(reply))
    assert b.source == "llm:fake/test"
    assert [s.text for s in b.sentences] == ["TCS reported an operating margin of 24.3% for the year."]
    assert b.dropped == 2                       # the wrong figure and the hedged speculation


def test_falls_back_to_verified_facts_when_the_model_has_nothing_usable(pipe, report):
    b = make_briefing(pipe, report, "q", FakeLLM("TCS will double."))
    assert b.source == "rules" and b.sentences


def test_rule_based_briefing_without_a_model(pipe, report):
    b = make_briefing(pipe, report, "q", None)
    assert b.source == "rules" and len(b.sentences) >= 1
    assert all(s.status != "contradicted" for s in b.sentences)


def test_chatbot_typography_is_normalised():
    assert normalise("the 20‑day average 【6】 and [3] done") == "the 20-day average  and  done"


def test_nonbreaking_hyphen_and_citation_marks_do_not_become_numbers(pipe):
    r = audit_text("TCS closed above its 20‑day average【6】.", pipe, ticker="TCS.NS")
    assert [n.raw for n in r.sentences[0].numbers] == []


def test_corrections_are_only_offered_when_the_data_confirms_them(pipe):
    r = audit_text("Operating margin for the year was 26.3%, a movement of -70 bps against "
                   "the prior year.", pipe, ticker="TCS.NS")
    s = r.sentences[0]
    assert s.status == "contradicted" and "24.3%" in s.explanation
    assert s.correction and "24.3%" in s.correction
    assert r.corrected_text == s.correction
    assert r.headline.startswith("0 of 1 checkable")
