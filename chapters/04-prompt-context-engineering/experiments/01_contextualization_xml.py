"""Phase 1: contextualization (XML-tag structuring) — full 1,001-row run, scored on the SAME 5
folds as Chapter 3's baselines (reuses cv_folds.json, never regenerated) so results are directly
comparable to Ch.3's zero-shot number.

Same structure as Chapter 3's 04_llm_classifier.py (row-level resumable — this environment has a
documented history of killing long-running background processes for reasons outside the code,
Ollama itself stays healthy per Ch.3's own notes), but logs the FULL timing_ms() breakdown per row
instead of only eval_duration, fixing the gap that made Ch.3's own "50 min" figure decode-only.

Prompt wording is frozen (prompts.py::build_prompt_xml) — not tuned against this run's results.

Run: uv run python chapters/04-prompt-context-engineering/experiments/01_contextualization_xml.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

from config import MODEL, NUM_CTX, TEMPERATURE
from prompts import build_prompt_xml

from cip.evals.golden_set import load_golden_set
from cip.evals.metrics import score_binary
from cip.llm.ollama_client import generate
from cip.llm.timing import timing_ms

HERE = Path(__file__).parent
CH3_EXPERIMENTS = HERE.parent.parent / "03-eval-harness" / "experiments"
GOLDEN_SET_PATH = CH3_EXPERIMENTS / "golden_set_labeled.csv"
FOLDS_PATH = CH3_EXPERIMENTS / "cv_folds.json"
RESULTS_PATH = HERE / "results_01_contextualization_xml.json"
PARTIAL_PATH = HERE / "results_01_contextualization_xml_partial.json"

RETRY_ATTEMPTS = 3
RETRY_DELAY_S = 8


def classify_with_retry(narrative: str) -> tuple[int | None, str, dict]:
    prompt, schema = build_prompt_xml(narrative)
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            result = generate(
                MODEL, prompt, options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
                format=schema,
            )
            break
        except requests.exceptions.RequestException as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"    [retry {attempt}/{RETRY_ATTEMPTS - 1}] {exc!r}", flush=True)
            time.sleep(RETRY_DELAY_S)

    timing = timing_ms(result)
    try:
        parsed = json.loads(result.response)
        pred = int(bool(parsed["churn_signal"]))
        rationale = str(parsed.get("rationale", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        pred = None
        rationale = ""
    return pred, rationale, timing


def load_partial() -> dict:
    if PARTIAL_PATH.exists():
        return json.loads(PARTIAL_PATH.read_text())
    return {}


def save_partial(partial: dict) -> None:
    PARTIAL_PATH.write_text(json.dumps(partial, indent=2))


def main() -> int:
    if not GOLDEN_SET_PATH.exists():
        sys.exit(f"{GOLDEN_SET_PATH} missing")
    if not FOLDS_PATH.exists():
        sys.exit(f"{FOLDS_PATH} missing")

    df = load_golden_set(GOLDEN_SET_PATH)
    folds = json.loads(FOLDS_PATH.read_text())
    fold_by_id = folds["fold_by_complaint_id"]

    partial = load_partial()
    todo = df[~df["complaint_id"].astype(str).isin(partial.keys())]
    print(f"{len(partial)}/{len(df)} already classified, {len(todo)} remaining", flush=True)

    for i, (_, row) in enumerate(todo.iterrows()):
        cid = str(row["complaint_id"])
        pred, rationale, timing = classify_with_retry(row["narrative"])
        partial[cid] = {
            "product": row["product"],
            "true_label": int(row["churn_signal"]),
            "predicted_label": pred if pred is not None else 0,
            "parse_failed": pred is None,
            "rationale": rationale,
            **timing,
        }
        save_partial(partial)
        if (i + 1) % 10 == 0 or i == len(todo) - 1:
            print(f"  {len(partial)}/{len(df)} classified (saved)", flush=True)

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
    total_wall_seconds = sum(v["wall_seconds"] for v in partial.values())
    total_prefill_ms = sum(v["prompt_eval_duration_ms"] for v in partial.values())
    total_decode_ms = sum(v["eval_duration_ms"] for v in partial.values())

    results = {
        "model": MODEL,
        "num_ctx": NUM_CTX,
        "prompting": "contextualization (XML-tag structured, same substantive rubric as Ch.3 "
                     "zero-shot condition A)",
        "evaluation": "5-fold stratified cross-validation (same folds as Ch.3's 03_baseline_tfidf.py)",
        "n_total": len(df),
        "n_positive_total": int(df["churn_signal"].sum()),
        "n_parse_failures": n_parse_failures,
        "total_wall_seconds": round(total_wall_seconds, 1),
        "total_prefill_ms": round(total_prefill_ms, 1),
        "total_decode_ms": round(total_decode_ms, 1),
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
    print(f"total wall-clock: {total_wall_seconds / 60:.1f} minutes "
          f"(prefill {total_prefill_ms / 1000 / 60:.1f} min, decode {total_decode_ms / 1000 / 60:.1f} min)")
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
