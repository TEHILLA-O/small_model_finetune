import numpy as np

from smft.metrics import bootstrap_ci, classification_report, ece, mcnemar, paired_bootstrap_diff


def test_ece_zero_when_confidence_matches_accuracy():
    probs = np.array([0.9] * 10 + [0.1] * 10)
    labels = np.array([1] * 9 + [0] + [0] * 9 + [1])
    assert ece(probs, labels) < 1e-9


def test_classification_report_fields():
    rep = classification_report([0.9, 0.2, 0.7, 0.4], [1, 0, 0, 1])
    assert rep["accuracy"] == 0.5
    assert rep["confusion_matrix"] == [[1, 1], [1, 1]]


def test_mcnemar_identical_models():
    c = np.array([1, 0, 1, 1, 0])
    assert mcnemar(c, c)["p_value"] == 1.0


def test_mcnemar_detects_one_sided_disagreement():
    a = np.ones(100, dtype=int)
    b = np.r_[np.zeros(30, dtype=int), np.ones(70, dtype=int)]
    assert mcnemar(a, b)["p_value"] < 1e-6


def test_bootstrap_intervals_cover_point_estimates():
    c = np.random.default_rng(0).integers(0, 2, 400)
    lo, hi = bootstrap_ci(c, n_boot=300)
    assert lo <= c.mean() <= hi
    lo, hi = paired_bootstrap_diff(c, c, n_boot=100)
    assert lo == hi == 0.0
