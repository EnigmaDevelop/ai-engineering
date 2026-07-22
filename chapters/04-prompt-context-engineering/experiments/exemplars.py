"""Fold-safe few-shot exemplar selection — born in Chapter 4.

For a given CV fold, exemplars are drawn ONLY from that fold's TRAIN partition (rows NOT in that
fold's test set per cv_folds.json), never from the fold's own test rows — this is what keeps
few-shot scoring leak-free and equivalent to an out-of-fold evaluation (each row is scored exactly
once, using its own fold's exemplars; see ~/.claude/plans/ch4-prompt-context-engineering.md, Phase
3). Selection is a fixed, seeded draw, frozen before any few-shot result is seen.
"""

from __future__ import annotations

import pandas as pd

K_POSITIVE = 2
K_NEGATIVE = 2
RANDOM_STATE = 42


def _reference_rationale(row: pd.Series) -> str:
    """A short, human-written-style rationale for the exemplar's label — built from the golden
    set's own evidence_quote, NOT the LLM's own output (an LLM-generated exemplar rationale would
    make the exemplar's style depend on the model being evaluated)."""
    quote = row.get("evidence_quote")
    if row["churn_signal"] == 1 and isinstance(quote, str) and quote.strip():
        return f'states: "{quote}"'
    if row["churn_signal"] == 1:
        return "explicitly states account closure, a switch, or an intent to leave"
    return "no statement about ending the relationship"


def select_exemplars_per_fold(df: pd.DataFrame, folds: dict) -> dict[int, list[dict]]:
    """Returns {fold_idx: [4 exemplar dicts]} — 2 positive + 2 negative per fold, drawn from that
    fold's train partition only."""
    fold_by_id = folds["fold_by_complaint_id"]
    n_folds = folds["n_folds"]
    df = df.copy()
    df["complaint_id"] = df["complaint_id"].astype(str)
    df["fold"] = df["complaint_id"].map(fold_by_id)

    exemplars_by_fold: dict[int, list[dict]] = {}
    for fold_idx in range(n_folds):
        train_df = df[df["fold"] != fold_idx]
        pos = train_df[train_df["churn_signal"] == 1].sample(n=K_POSITIVE, random_state=RANDOM_STATE)
        neg = train_df[train_df["churn_signal"] == 0].sample(n=K_NEGATIVE, random_state=RANDOM_STATE)
        exemplars = [
            {
                "narrative": row["narrative"],
                "churn_signal": int(row["churn_signal"]),
                "rationale": _reference_rationale(row),
            }
            for _, row in pd.concat([pos, neg]).iterrows()
        ]
        exemplars_by_fold[fold_idx] = exemplars
    return exemplars_by_fold
