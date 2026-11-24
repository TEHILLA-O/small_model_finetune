"""Classification, calibration and significance metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error on the predicted class confidence."""
    conf = np.maximum(probs, 1 - probs)
    pred = (probs >= 0.5).astype(int)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi) if hi < 1 else (probs >= lo) & (probs <= hi)
        if mask.any():
            out.append(
                {
                    "bin_low": float(lo),
                    "bin_high": float(hi),
                    "count": int(mask.sum()),
                    "mean_predicted": float(probs[mask].mean()),
                    "observed_positive_rate": float(labels[mask].mean()),
                }
            )
    return out


def classification_report(probs, labels) -> dict:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (probs >= 0.5).astype(int)
    return {
        "n": int(len(labels)),
        "accuracy": float(accuracy_score(labels, pred)),
        "f1": float(f1_score(labels, pred)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "ece": ece(probs, labels),
        "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist(),
    }


def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(correct)
    stats = np.array([correct[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def paired_bootstrap_diff(a_correct, b_correct, n_boot: int = 2000, seed: int = 0):
    """95% CI for accuracy(a) minus accuracy(b) on the same examples."""
    rng = np.random.default_rng(seed)
    n = len(a_correct)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        diffs.append(a_correct[i].mean() - b_correct[i].mean())
    diffs = np.array(diffs)
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Exact McNemar test on the discordant pairs."""
    b = int(((a_correct == 1) & (b_correct == 0)).sum())
    c = int(((a_correct == 0) & (b_correct == 1)).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"a_right_b_wrong": b, "a_wrong_b_right": c, "p_value": float(p)}
