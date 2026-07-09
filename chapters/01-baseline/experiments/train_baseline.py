"""Baseline churn model: XGBoost on the Telco Customer Churn table.

Every later chapter's LLM-based approach gets measured against this. Loads directly from the
DuckDB warehouse built in Ch.0 (no CSV re-read), does the minimum real cleaning the dataset
needs, and reports both a trivial majority-class baseline and the XGBoost model so the lift is
an honest, measured number rather than an assumption.

Usage:
    uv run python chapters/01-baseline/experiments/train_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"
RESULTS_PATH = Path(__file__).resolve().parent / "results.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

CATEGORICAL_COLS = [
    "gender", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaymentMethod",
]
BOOLEAN_COLS = ["SeniorCitizen", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def load_data() -> pd.DataFrame:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute("SELECT * FROM telco_customers").fetchdf()
    finally:
        con.close()
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.drop(columns=["customerID"]).copy()

    # 11 rows have blank TotalCharges, all with tenure == 0 (brand-new customers, no bill
    # issued yet) — confirmed by direct query, not assumed. 0.0 is the correct value, not an
    # imputation guess: tenure * MonthlyCharges == 0 for these rows.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    df["SeniorCitizen"] = df["SeniorCitizen"].astype(bool)
    for col in BOOLEAN_COLS:
        df[col] = df[col].astype(int)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    y = df["Churn"].astype(int)
    X = df.drop(columns=["Churn"])
    return X, y


def majority_class_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    return score(y_test, y_pred, y_proba)


def score(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
    }


def main() -> int:
    if not DB_PATH.exists():
        sys.exit(f"{DB_PATH} missing — run `uv run python -m cip.data.ingest` first (Ch.0)")

    df = load_data()
    X, y = clean(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=class_ratio,
        tree_method="hist",
        enable_categorical=True,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_roc_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    xgb_metrics = score(y_test, y_pred, y_proba)
    dummy_metrics = majority_class_baseline(X_train, y_train, X_test, y_test)

    cm = confusion_matrix(y_test, y_pred).tolist()

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )

    results = {
        "dataset": {
            "n_rows": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "churn_rate_overall": float(y.mean()),
            "churn_rate_train": float(y_train.mean()),
            "churn_rate_test": float(y_test.mean()),
        },
        "model": {
            "type": "xgboost.XGBClassifier",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "scale_pos_weight": float(class_ratio),
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "roc_auc_mean": float(cv_roc_auc.mean()),
            "roc_auc_std": float(cv_roc_auc.std()),
            "f1_mean": float(cv_f1.mean()),
            "f1_std": float(cv_f1.std()),
        },
        "test_set_metrics": {
            "xgboost": xgb_metrics,
            "majority_class_baseline": dummy_metrics,
        },
        "confusion_matrix": {
            "labels": ["no_churn", "churn"],
            "matrix": cm,
        },
        "top_10_feature_importance": importances.round(4).to_dict(),
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print(f"n={len(df)} train={len(X_train)} test={len(X_test)} churn_rate={y.mean():.3f}")
    print(f"CV (train, {CV_FOLDS}-fold): roc_auc={cv_roc_auc.mean():.4f}+-{cv_roc_auc.std():.4f} "
          f"f1={cv_f1.mean():.4f}+-{cv_f1.std():.4f}")
    print("Test set — XGBoost:", {k: round(v, 4) for k, v in xgb_metrics.items()})
    print("Test set — majority-class baseline:", {k: round(v, 4) for k, v in dummy_metrics.items()})
    print("Confusion matrix [no_churn, churn]:", cm)
    print("Top 10 feature importances:", importances.round(4).to_dict())
    print(f"\nResults written to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
