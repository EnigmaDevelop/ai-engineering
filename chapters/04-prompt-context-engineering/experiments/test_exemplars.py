"""Unit tests for fold-safe few-shot exemplar selection (exemplars.py)."""

import pandas as pd

from exemplars import select_exemplars_per_fold


def make_df(n_per_fold: int = 20, n_folds: int = 3) -> tuple[pd.DataFrame, dict]:
    rows = []
    fold_by_complaint_id = {}
    cid = 0
    for fold_idx in range(n_folds):
        for i in range(n_per_fold):
            cid += 1
            rows.append({
                "complaint_id": cid,
                "narrative": f"narrative {cid}",
                "churn_signal": 1 if i < 5 else 0,
                "evidence_quote": "closed my account" if i < 5 else None,
            })
            fold_by_complaint_id[str(cid)] = fold_idx
    df = pd.DataFrame(rows)
    folds = {"n_folds": n_folds, "fold_by_complaint_id": fold_by_complaint_id}
    return df, folds


def test_returns_one_entry_per_fold():
    df, folds = make_df()
    exemplars_by_fold = select_exemplars_per_fold(df, folds)
    assert set(exemplars_by_fold.keys()) == {0, 1, 2}


def test_each_fold_gets_two_positive_two_negative():
    df, folds = make_df()
    exemplars_by_fold = select_exemplars_per_fold(df, folds)
    for exemplars in exemplars_by_fold.values():
        assert len(exemplars) == 4
        labels = [e["churn_signal"] for e in exemplars]
        assert labels.count(1) == 2
        assert labels.count(0) == 2


def test_exemplars_never_come_from_their_own_fold():
    df, folds = make_df()
    exemplars_by_fold = select_exemplars_per_fold(df, folds)
    fold_by_id = folds["fold_by_complaint_id"]
    narrative_to_id = {row["narrative"]: str(row["complaint_id"]) for _, row in df.iterrows()}

    for fold_idx, exemplars in exemplars_by_fold.items():
        for ex in exemplars:
            source_id = narrative_to_id[ex["narrative"]]
            assert fold_by_id[source_id] != fold_idx


def test_positive_rationale_uses_evidence_quote():
    df, folds = make_df()
    exemplars_by_fold = select_exemplars_per_fold(df, folds)
    positive_exemplars = [e for exs in exemplars_by_fold.values() for e in exs if e["churn_signal"] == 1]
    assert all("closed my account" in e["rationale"] for e in positive_exemplars)
