"""TF-IDF + Logistic Regression baseline for churn-signal classification.

The "20-year-old algorithm" baseline every LLM approach in this chapter is measured against —
same train/test discipline as Ch.1's XGBoost baseline (stratified, fixed random_state), now using
**5-fold stratified cross-validation** rather than a single 70/30 split. With only 48 positives in
1001 rows (4.8%), a single split holds out ~10 positives — one misclassified example swings
recall by ~10 points, a false precision. 5-fold CV scores every row exactly once as a held-out
test case across the 5 folds, so all 48 positives contribute to the reported metrics, and the
fold-to-fold spread (reported as mean ± std) shows how much the result actually varies — the
point of this chapter is measuring that variance, not hiding it behind a single lucky/unlucky
split. Trained and scored on the human-verified golden set (golden_set_labeled.csv) — ambiguous
rows are dropped by cip.evals.golden_set.load_golden_set (none in this set).

The fold assignment (by complaint_id) is saved to cv_folds.json so 04_llm_classifier.py scores
on the EXACT same folds — apples-to-apples, not two independently drawn samples.

Run: uv run python chapters/03-eval-harness/experiments/03_baseline_tfidf.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from cip.evals.golden_set import load_golden_set
from cip.evals.metrics import score_binary

HERE = Path(__file__).parent
GOLDEN_SET_PATH = HERE / "golden_set_labeled.csv"
FOLDS_PATH = HERE / "cv_folds.json"
RESULTS_PATH = HERE / "results_baseline_tfidf.json"

RANDOM_STATE = 42
N_FOLDS = 5
TOP_N_TERMS = 10


def mean_std(values: list[float]) -> dict:
    arr = np.array(values, dtype=float)
    return {"mean": round(float(arr.mean()), 3), "std": round(float(arr.std()), 3)}


def main() -> int:
    if not GOLDEN_SET_PATH.exists():
        sys.exit(f"{GOLDEN_SET_PATH} missing — run label_cli.py first")

    df = load_golden_set(GOLDEN_SET_PATH).reset_index(drop=True)
    n_ambiguous = df.attrs.get("n_ambiguous_dropped", 0)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_assignment: dict[str, int] = {}
    per_fold_metrics = []
    all_predictions: dict[str, int] = {}

    for fold_idx, (train_pos, test_pos) in enumerate(skf.split(df, df["churn_signal"])):
        train_df = df.iloc[train_pos]
        test_df = df.iloc[test_pos]
        for cid in test_df["complaint_id"]:
            fold_assignment[str(cid)] = fold_idx

        # class_weight="balanced" because positives are ~4.8% of the golden set — same
        # imbalance-handling discipline as Ch.1's XGBoost scale_pos_weight, applied to LR here.
        vectorizer = TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2), min_df=2, stop_words="english"
        )
        X_train = vectorizer.fit_transform(train_df["narrative"])
        X_test = vectorizer.transform(test_df["narrative"])

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        clf.fit(X_train, train_df["churn_signal"])
        y_pred = clf.predict(X_test)

        fold_metrics = score_binary(test_df["churn_signal"].tolist(), y_pred.tolist())
        fold_metrics["n_test"] = len(test_df)
        fold_metrics["n_positive_test"] = int(test_df["churn_signal"].sum())
        per_fold_metrics.append(fold_metrics)

        for cid, pred in zip(test_df["complaint_id"], y_pred, strict=True):
            all_predictions[str(cid)] = int(pred)

        print(f"fold {fold_idx}: n_test={len(test_df)} "
              f"n_positive_test={fold_metrics['n_positive_test']} "
              f"f1={fold_metrics['f1']:.3f}")

    FOLDS_PATH.write_text(json.dumps({
        "n_folds": N_FOLDS,
        "random_state": RANDOM_STATE,
        "fold_by_complaint_id": fold_assignment,
    }, indent=2))

    aggregated = {
        metric: mean_std([fold[metric] for fold in per_fold_metrics])
        for metric in ("accuracy", "precision", "recall", "f1")
    }

    # Separate final fit on ALL data, only for interpretability (top terms) — never used for
    # scoring, since that would leak train/test information into the reported metrics.
    full_vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, stop_words="english"
    )
    X_full = full_vectorizer.fit_transform(df["narrative"])
    full_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    full_clf.fit(X_full, df["churn_signal"])
    feature_names = full_vectorizer.get_feature_names_out()
    coef = full_clf.coef_[0]
    top_idx = coef.argsort()[::-1][:TOP_N_TERMS]
    top_terms = [{"term": feature_names[i], "coef": round(float(coef[i]), 3)} for i in top_idx]

    results = {
        "model": "TfidfVectorizer + LogisticRegression",
        "evaluation": "5-fold stratified cross-validation",
        "n_total": len(df),
        "n_ambiguous_dropped": n_ambiguous,
        "n_positive_total": int(df["churn_signal"].sum()),
        "n_folds": N_FOLDS,
        "aggregated_metrics": aggregated,
        "per_fold_metrics": per_fold_metrics,
        "top_positive_terms_full_fit": top_terms,
        "predictions": all_predictions,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print("\naggregated (mean ± std across 5 folds):")
    print(json.dumps(aggregated, indent=2))
    print(f"top terms (full-data fit, interpretability only): {[t['term'] for t in top_terms]}")
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
