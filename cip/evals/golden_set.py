"""Golden-set loading/validation for the eval harness. Born in Chapter 3.

The golden set is a small, real, human-verified CSV (never synthetic). Labels are proposed by
Claude (`claude -p`, subscription auth) and every row is reviewed by the human annotator — LLM
proposals never become golden labels without human sign-off, because a golden set labeled solely
by an LLM would make every later LLM evaluation circular (agreement with the labeler, not
accuracy). See chapters/03-eval-harness/experiments/ for the sampling, pre-labeling, and review
scripts and the labeling rubric.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["complaint_id", "product", "narrative", "churn_signal"]


def load_golden_set(path: Path) -> pd.DataFrame:
    """Load a human-verified golden set. Rows the annotator marked ambiguous are dropped
    (their count is worth reporting — do it at the call site via the returned attrs)."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    n_ambiguous = 0
    if "ambiguous" in df.columns:
        ambiguous_mask = df["ambiguous"].fillna(False).astype(bool)
        n_ambiguous = int(ambiguous_mask.sum())
        df = df[~ambiguous_mask].copy()

    if df["churn_signal"].isna().any():
        n_unlabeled = int(df["churn_signal"].isna().sum())
        raise ValueError(
            f"{path} has {n_unlabeled} unlabeled non-ambiguous rows — finish the review pass "
            "(label_cli.py) before loading"
        )
    df["churn_signal"] = df["churn_signal"].astype(int)
    if not df["churn_signal"].isin([0, 1]).all():
        raise ValueError(f"{path}: churn_signal must be 0 or 1 for every non-ambiguous row")

    df.attrs["n_ambiguous_dropped"] = n_ambiguous
    return df
