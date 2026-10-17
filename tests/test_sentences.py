from agents.analysts import first_sentence


def test_decimals_and_abbreviations_do_not_end_a_sentence():
    assert first_sentence("ITC Limited has acquired the remaining 52.4% stake. Shares rose.") \
        == "ITC Limited has acquired the remaining 52.4% stake."
    assert first_sentence("Revenue grew 6.8% over the previous year. Margins fell.") \
        == "Revenue grew 6.8% over the previous year."
    assert first_sentence("Profit was ₹1,234.5 cr in Q4. Next quarter is later.") \
        == "Profit was ₹1,234.5 cr in Q4."


def test_text_without_a_final_stop_gets_one():
    assert first_sentence("Shares advanced after strong results") == "Shares advanced after strong results."
    assert first_sentence("Shares advanced.") == "Shares advanced."
    assert first_sentence("Is it cheap? Maybe.") == "Is it cheap?"


def test_the_fundamental_revision_quotes_a_whole_sentence():
    """Regression: revise() used split('.')[0], which cut '6.8%' down to '6'."""
    from agents.analysts import FundamentalAgent
    from core import Claim
    from pipeline import ResearchPipeline
    ctx = ResearchPipeline(prefer_live=False)._prepare("TCS.NS")
    agent = FundamentalAgent(ctx["store"])
    revised = agent.revise(Claim.new("fundamental", "s", "revenue growth over previous year"),
                           "x", "TCS.NS", ctx)
    assert revised is not None and revised.text.endswith("previous financial year.")
