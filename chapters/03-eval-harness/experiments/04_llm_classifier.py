"""Zero-shot local LLM classifier for churn-signal — scored on the SAME 5 folds as
03_baseline_tfidf.py (reads cv_folds.json), so the TF-IDF-vs-LLM comparison is apples-to-apples.

Zero-shot classification has no "training" step, so the LLM classifies every row in the golden
set exactly once (fold membership can't leak into a zero-shot call — there's nothing to leak
into). The fold assignment is used only afterward, to aggregate metrics the same way as the
TF-IDF baseline (mean ± std across the same 5 partitions) — same evaluation methodology, so the
two numbers are directly comparable.

Deliberately zero-shot only: few-shot / CoT prompting is explicitly Ch.4's job (per the
roadmap, Ch.4's experiments run "all measured with Ch.3's harness") — doing it here would blur
the chapter boundary.

Model is llama3.2:3b (project default per ADR-0001 hardware constraints), NOT Claude — Claude
pre-labeled the golden set in 02_prelabel_claude.py, so evaluating Claude against these labels
would be partially circular even with human sign-off. Structured output via format=<schema>,
same pattern as Ch.2's 03_structured_output.py.

Row-level resumable (checkpoints after every single call to results_llm_classifier_partial.json)
— 1001 narratives on CPU-bound hardware (ADR-0001) is a long run, and this project has already
hit interruptions (Claude Pro session limits, background-task terminations) during long local runs.

Run: uv run python chapters/03-eval-harness/experiments/04_llm_classifier.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

from cip.evals.golden_set import load_golden_set
from cip.evals.metrics import score_binary
from cip.llm.ollama_client import generate

HERE = Path(__file__).parent
GOLDEN_SET_PATH = HERE / "golden_set_labeled.csv"
FOLDS_PATH = HERE / "cv_folds.json"
RESULTS_PATH = HERE / "results_llm_classifier.json"
PARTIAL_PATH = HERE / "results_llm_classifier_partial.json"

MODEL = "llama3.2:3b"
RETRY_ATTEMPTS = 3
RETRY_DELAY_S = 8

SCHEMA = {
    "type": "object",
    "properties": {
        "churn_signal": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["churn_signal", "rationale"],
}

RUBRIC = (
    "You classify US consumer complaint narratives for an EXPLICIT signal that the consumer "
    "is ending their relationship with the company.\n\n"
    "churn_signal = true ONLY if the narrative explicitly states the consumer closed / is "
    "closing the account, switched / is switching to another provider, or states a definite "
    "intention or threat to leave because of the issue described.\n"
    "churn_signal = false otherwise — complaining, disputing, anger, but no statement about "
    "ending the relationship."
)


def build_prompt(narrative: str) -> str:
    return (
        f"{RUBRIC}\n\nNARRATIVE:\n{narrative}\n\n"
        "Return JSON with your classification and a one-sentence rationale: if true, quote the "
        "relevant part of the narrative; if false, briefly say why no such statement exists."
    )


def classify_with_retry(narrative: str) -> tuple[int | None, str, float]:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            result = generate(
                MODEL, build_prompt(narrative), options={"temperature": 0}, format=SCHEMA
            )
            break
        except requests.exceptions.RequestException as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"    [retry {attempt}/{RETRY_ATTEMPTS - 1}] {exc!r}", flush=True)
            time.sleep(RETRY_DELAY_S)

    eval_duration_ms = round(result.eval_duration_ns / 1e6, 1)
    try:
        parsed = json.loads(result.response)
        pred = int(bool(parsed["churn_signal"]))
        rationale = str(parsed.get("rationale", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        pred = None
        rationale = ""
    return pred, rationale, eval_duration_ms


def load_partial() -> dict:
    if PARTIAL_PATH.exists():
        return json.loads(PARTIAL_PATH.read_text())
    return {}


def save_partial(partial: dict) -> None:
    PARTIAL_PATH.write_text(json.dumps(partial, indent=2))


def main() -> int:
    if not GOLDEN_SET_PATH.exists():
        sys.exit(f"{GOLDEN_SET_PATH} missing — run label_cli.py first")
    if not FOLDS_PATH.exists():
        sys.exit(f"{FOLDS_PATH} missing — run 03_baseline_tfidf.py first")

    df = load_golden_set(GOLDEN_SET_PATH)
    folds = json.loads(FOLDS_PATH.read_text())
    fold_by_id = folds["fold_by_complaint_id"]

    partial = load_partial()
    todo = df[~df["complaint_id"].astype(str).isin(partial.keys())]
    print(f"{len(partial)}/{len(df)} already classified, {len(todo)} remaining", flush=True)

    for i, (_, row) in enumerate(todo.iterrows()):
        cid = str(row["complaint_id"])
        pred, rationale, eval_ms = classify_with_retry(row["narrative"])
        partial[cid] = {
            "product": row["product"],
            "true_label": int(row["churn_signal"]),
            "predicted_label": pred if pred is not None else 0,
            "parse_failed": pred is None,
            "rationale": rationale,
            "eval_duration_ms": eval_ms,
        }
        save_partial(partial)
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            print(f"  {len(partial)}/{len(df)} classified (saved)", flush=True)

    # Aggregate per fold, same structure as 03_baseline_tfidf.py, for direct comparison.
    n_folds = folds["n_folds"]
    per_fold_metrics = []
    for fold_idx in range(n_folds):
        fold_ids = [cid for cid, f in fold_by_id.items() if f == fold_idx]
        y_true = [partial[cid]["true_label"] for cid in fold_ids]
        y_pred = [partial[cid]["predicted_label"] for cid in fold_ids]
        fold_metrics = score_binary(y_true, y_pred)
        fold_metrics["n_test"] = len(fold_ids)
        fold_metrics["n_positive_test"] = sum(y_true)
        per_fold_metrics.append(fold_metrics)
        print(f"fold {fold_idx}: n_test={len(fold_ids)} "
              f"n_positive_test={fold_metrics['n_positive_test']} f1={fold_metrics['f1']:.3f}")

    def mean_std(values: list[float]) -> dict:
        arr = np.array(values, dtype=float)
        return {"mean": round(float(arr.mean()), 3), "std": round(float(arr.std()), 3)}

    aggregated = {
        metric: mean_std([fold[metric] for fold in per_fold_metrics])
        for metric in ("accuracy", "precision", "recall", "f1")
    }

    n_parse_failures = sum(1 for v in partial.values() if v["parse_failed"])
    total_eval_ms = sum(v["eval_duration_ms"] for v in partial.values())

    results = {
        "model": MODEL,
        "prompting": "zero-shot",
        "evaluation": "5-fold stratified cross-validation (same folds as 03_baseline_tfidf.py)",
        "n_total": len(df),
        "n_positive_total": int(df["churn_signal"].sum()),
        "n_parse_failures": n_parse_failures,
        "total_eval_duration_ms": round(total_eval_ms, 1),
        "aggregated_metrics": aggregated,
        "per_fold_metrics": per_fold_metrics,
        "rows": partial,
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    if PARTIAL_PATH.exists():
        PARTIAL_PATH.unlink()

    print("\naggregated (mean ± std across 5 folds):")
    print(json.dumps(aggregated, indent=2))
    print(f"parse_failures: {n_parse_failures}/{len(df)}")
    print(f"total eval time: {total_eval_ms / 1000 / 60:.1f} minutes")
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
