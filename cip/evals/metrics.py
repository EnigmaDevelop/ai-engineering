"""Shared binary-classification scoring. Born in Chapter 3, reused by Ch.4+.

Thin wrapper over sklearn.metrics (already a project dependency, used the same way in Ch.1's
baseline) — not a reimplementation. The point of this module is one consistent scoring function
every eval-harness experiment calls, not new metric math.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def score_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """y_true/y_pred: any sequence of 0/1 ints. Returns accuracy/precision/recall/f1 plus a
    confusion matrix as [[tn, fp], [fn, tp]]."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
