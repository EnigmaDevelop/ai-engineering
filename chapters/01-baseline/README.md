# Chapter 1 — Baseline first

Every LLM-based approach in this repo (Ch.3 onward) gets measured against this model. If an LLM
can't beat a boring gradient-boosted tree on tabular data it had no business being asked to model
in the first place, that's the honest headline — not something to bury.

## Goal

A properly-validated XGBoost churn classifier on the Telco Customer Churn table, plus a trivial
majority-class baseline in the same run — so every metric reported is a *lift over doing nothing*,
not a number in isolation.

## Method

- **Source:** `telco_customers` table in `data/warehouse.duckdb` (built in Ch.0), read directly via
  DuckDB — no intermediate CSV re-read.
- **Cleaning:** `customerID` dropped (identifier, not a feature). `TotalCharges` is stored as
  `VARCHAR` in the raw CSV; 11 rows have a blank value, and querying them directly confirms all 11
  have `tenure == 0` (brand-new customers, no bill issued yet) — so they're filled with `0.0`, which
  is the correct value (`tenure * MonthlyCharges == 0`), not an imputation guess.
- **Categoricals:** the 11 nominal columns (`Contract`, `InternetService`, `PaymentMethod`, etc.)
  are cast to pandas `category` dtype and handled by XGBoost's native categorical split support
  (`enable_categorical=True`, `tree_method="hist"`) — no one-hot encoding step.
- **Split:** stratified 80/20 train/test (`random_state=42`), so the ~26.5% churn rate is preserved
  in both halves (confirmed in `results.json`: 26.54% train vs. 26.54% test).
- **Class imbalance:** `scale_pos_weight` set to the train set's negative/positive ratio (2.77),
  not left at XGBoost's default of 1 — churn is the minority class and recall on it is the point of
  the model.
- **Validation discipline:** 5-fold stratified cross-validation on the training set only (reported
  as mean ± std, not a single number) to check the model isn't just lucky on one split, *then* a
  single fit evaluated once on the untouched test set for the numbers that count.
- **Comparison baseline:** `sklearn.dummy.DummyClassifier(strategy="most_frequent")` run through the
  identical train/test split and metric set — this is what "the model" has to beat, concretely.

Script: [`experiments/train_baseline.py`](experiments/train_baseline.py). Raw results:
[`experiments/results.json`](experiments/results.json).

## Results

**Measured 2026-07-09**, `n=7,043` (train 5,634 / test 1,409), churn rate 26.5%.

5-fold CV on the training set: ROC-AUC 0.8439 ± 0.0098, F1 0.6315 ± 0.0117 — stable across folds,
not a single-split fluke.

| Metric | Majority-class baseline | XGBoost |
|---|---|---|
| Accuracy | 0.7346 | 0.7551 |
| Precision | 0.0000 | 0.5264 |
| Recall | 0.0000 | 0.7727 |
| F1 | 0.0000 | 0.6262 |
| ROC-AUC | 0.5000 | 0.8405 |
| PR-AUC | 0.2654 | 0.6545 |
| Brier score (lower is better) | 0.2654 | 0.1608 |

Confusion matrix (test set, XGBoost):

|  | Predicted no-churn | Predicted churn |
|---|---|---|
| **Actual no-churn** | 775 | 260 |
| **Actual churn** | 85 | 289 |

Top-10 feature importances (gain): `Contract` 0.314, `InternetService` 0.138, `OnlineSecurity`
0.119, `TechSupport` 0.063, `StreamingMovies` 0.039, `tenure` 0.034, `PaymentMethod` 0.030,
`PhoneService` 0.028, `StreamingTV` 0.027, `PaperlessBilling` 0.025.

## Decision

- **Accuracy alone would have been misleading here on purpose left unexamined**: the majority-class
  baseline scores 73.5% accuracy by predicting "no churn" for every single customer, while having
  zero precision, zero recall, and ROC-AUC exactly 0.5 — i.e., no discriminative power whatsoever.
  Any later chapter (or article) that reports only accuracy on this dataset without also reporting
  ROC-AUC/PR-AUC/recall is not reporting a real result. This is the anchor for that standard.
- **XGBoost's real lift is in ROC-AUC (0.50 → 0.84) and recall (0.00 → 0.77) on the churn class**,
  not in accuracy (which only moves 2 points) — accuracy is the wrong headline metric for a
  26.5%-imbalanced target, and this run makes that concrete rather than asserted.
- **Feature importance matches domain-known churn drivers** (contract length, security/support
  add-ons, internet service type) rather than an arbitrary or spurious signal — a real sanity check
  that the model learned something explicable, not noise.
- `scale_pos_weight` is a deliberate choice, not the XGBoost default: without it, the model would
  optimize for the majority class the same way the dummy baseline does, undermining the whole point
  of building a churn model in the first place.
- No hyperparameter search was run — `n_estimators=300`, `max_depth=4`, `learning_rate=0.05` are
  reasonable, commonly-used defaults for a dataset this size, not a tuned result. This is
  intentional: the goal of this chapter is an honest, reproducible anchor, not a leaderboard score.
  Revisit only if a later chapter's comparison hinges on squeezing more out of this model.

## Checklist

- [x] Load `telco_customers` from the Ch.0 DuckDB warehouse (no CSV re-read)
- [x] Document and resolve the `TotalCharges` blank-value data quality issue with evidence, not
      an assumption
- [x] Stratified train/test split + stratified 5-fold CV on the training set
- [x] Majority-class baseline reported alongside XGBoost, same split, same metrics
- [x] Full metric set: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, confusion
      matrix, feature importance
- [x] Results committed as JSON under `experiments/`
- [x] `uv run pytest` / `uv run ruff check .` clean (verified 2026-07-09: ruff `All checks passed!`;
      pytest collects 0 items — no test suites exist yet, same expected state as Ch.0)
