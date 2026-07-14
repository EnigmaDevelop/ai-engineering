"""Regenerates the 4 Kaggle dataset CSVs from Ch.3's already-committed experiment artifacts.

No recomputation, no LLM/API calls — every value here was already measured and committed in
chapters/03-eval-harness/. This script only reshapes existing results into a publishable,
self-contained tabular form (see ADR-0002).

Run: uv run python kaggle/churn-signal-golden-set/build_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "chapters" / "03-eval-harness" / "experiments"
OUT_DIR = Path(__file__).parent / "dataset"

GOLDEN_SET_PATH = EXPERIMENTS / "golden_set_labeled.csv"
LLM_RESULTS_PATH = EXPERIMENTS / "results_llm_classifier.json"
TFIDF_RESULTS_PATH = EXPERIMENTS / "results_baseline_tfidf.json"
FOLDS_PATH = EXPERIMENTS / "cv_folds.json"

N_ROWS_EXPECTED = 1001
N_POSITIVE_EXPECTED = 48
N_FOLDS_EXPECTED = 5


def build_golden_set() -> pd.DataFrame:
    df = pd.read_csv(GOLDEN_SET_PATH)
    df.to_csv(OUT_DIR / "golden_set.csv", index=False)
    return df


def build_cv_folds() -> dict[str, int]:
    folds = json.loads(FOLDS_PATH.read_text())
    fold_by_id: dict[str, int] = folds["fold_by_complaint_id"]
    out = pd.DataFrame(
        {"complaint_id": [int(cid) for cid in fold_by_id], "fold": list(fold_by_id.values())}
    ).sort_values("complaint_id")
    out.to_csv(OUT_DIR / "cv_folds.csv", index=False)
    return fold_by_id


def build_llm_predictions(fold_by_id: dict[str, int]) -> None:
    results = json.loads(LLM_RESULTS_PATH.read_text())
    rows = []
    for cid, row in results["rows"].items():
        rows.append({
            "complaint_id": int(cid),
            "product": row["product"],
            "fold": fold_by_id[cid],
            "true_label": row["true_label"],
            "predicted_label": row["predicted_label"],
            "parse_failed": row["parse_failed"],
            "rationale": row["rationale"],
            "eval_duration_ms": row["eval_duration_ms"],
        })
    out = pd.DataFrame(rows).sort_values("complaint_id")
    out.to_csv(OUT_DIR / "llm_predictions.csv", index=False)


def build_tfidf_predictions(fold_by_id: dict[str, int]) -> None:
    results = json.loads(TFIDF_RESULTS_PATH.read_text())
    predictions: dict[str, int] = results["predictions"]
    out = pd.DataFrame({
        "complaint_id": [int(cid) for cid in predictions],
        "fold": [fold_by_id[cid] for cid in predictions],
        "predicted_label": list(predictions.values()),
    }).sort_values("complaint_id")
    out.to_csv(OUT_DIR / "tfidf_predictions.csv", index=False)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    golden_df = build_golden_set()
    fold_by_id = build_cv_folds()
    build_llm_predictions(fold_by_id)
    build_tfidf_predictions(fold_by_id)

    assert len(golden_df) == N_ROWS_EXPECTED, f"golden_set.csv has {len(golden_df)} rows, expected {N_ROWS_EXPECTED}"
    assert int(golden_df["churn_signal"].sum()) == N_POSITIVE_EXPECTED, "positive count mismatch"
    assert golden_df["churn_signal"].isin([0, 1]).all(), "churn_signal must be 0/1"
    assert len(fold_by_id) == N_ROWS_EXPECTED, f"cv_folds has {len(fold_by_id)} rows, expected {N_ROWS_EXPECTED}"
    assert set(fold_by_id.values()) == set(range(N_FOLDS_EXPECTED)), "fold values must be 0..4"

    for name in ("llm_predictions.csv", "tfidf_predictions.csv"):
        check = pd.read_csv(OUT_DIR / name)
        assert len(check) == N_ROWS_EXPECTED, f"{name} has {len(check)} rows, expected {N_ROWS_EXPECTED}"
        assert not check["predicted_label"].isna().any(), f"{name} has NaN predicted_label"
        assert not check["fold"].isna().any(), f"{name} has NaN fold"

    print(f"wrote 4 CSVs to {OUT_DIR}")
    print(f"golden_set: {len(golden_df)} rows, {N_POSITIVE_EXPECTED} positive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
