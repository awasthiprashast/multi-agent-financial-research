"""The Streamlit app, run headlessly (offline snapshot data, no language model)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "frontend" / "streamlit_app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    return at


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_all_four_tabs_render(app):
    assert [t.label for t in app.tabs] == ["Company briefing", "Fact-check a text",
                                          "Chatbot vs the data", "How accurate is it?"]


def test_the_briefing_shows_figures_and_a_summary(app):
    _button(app, "Get briefing").click().run()
    assert not app.exception, app.exception
    assert any("flex-wrap" in m.value and "Last price" in m.value for m in app.markdown)
    assert any(m.value.startswith("- ") for m in app.markdown)


def test_fact_check_marks_a_wrong_figure_and_offers_the_corrected_text(app):
    app.text_area(key="audit_text").set_value(
        "TCS operating margin for the year was 99.9%, a movement of -70 bps against the prior year."
    ).run()
    _button(app, "Check it").click().run()
    assert not app.exception, app.exception
    assert any("contradict" in w.value or "do not" in w.value for w in app.warning)
    assert len(app.code) == 1                         # the corrected version


def test_the_accuracy_tab_publishes_the_measured_results(app):
    assert any("Real chatbot text" in m.value for m in app.markdown)
    assert any("Held-out result" in c.value for c in app.caption)
