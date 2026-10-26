"""Regression guard for the real-text benchmark (offline, frozen data).

The floors sit a little below the measured results so that a genuine regression fails the build
but harmless drift does not. They are not the headline numbers: see results/real_text_benchmark*.json.
"""

import pytest

from benchmark.labels import DEV, LABELS, TEST
from eval import benchmark_real as b


@pytest.fixture(scope="module")
def rows():
    return b.run(DEV + TEST)


def test_every_answer_has_one_label_per_sentence(rows):
    assert len(rows) == sum(len(c) for c in LABELS.values())


def test_the_auditor_keeps_its_measured_accuracy(rows):
    m = b.metrics(rows)
    assert m["errors_caught"]["recall"] >= 0.80
    assert m["errors_caught"]["precision"] >= 0.90
    assert m["false_reject_rate"] <= 0.03
    assert m["false_accept_rate"] <= 0.06
    assert m["accuracy_when_committed"] >= 0.95
    assert m["coverage_of_checkable"] >= 0.85


def test_a_verifier_that_trusts_numbers_does_far_worse(rows):
    assert b.naive_baseline(rows)["errors_caught_recall"] == 0.0
