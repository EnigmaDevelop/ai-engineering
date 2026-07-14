"""Cost comparison: TF-IDF+LogisticRegression vs local LLM zero-shot classifier.

Reuses timing already recorded by 03_baseline_tfidf.py's fold loop is NOT stored per-run, so this
script re-times ONE fresh fit+predict cycle (fold 0 of the same cv_folds.json split) to get a real
wall-clock number for the classical baseline — cheap (seconds), no need to redo the full 5-fold run.

LLM timing reuses results_llm_classifier.json's per-row eval_duration_ms (already measured during
the real classification run — nothing to redo there). Token counts were NOT recorded per-row
during that run, so this script makes a small, explicitly-labeled supplementary measurement: it
re-sends a random sample of N_TOKEN_SAMPLE narratives through the exact same prompt/schema with
num_predict capped, reading prompt_eval_count/eval_count back from Ollama — a sampled estimate,
not a full-corpus count (evidence-standard: labeled as such, not presented as exact).

Run: uv run python chapters/03-eval-harness/experiments/06_cost_comparison.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from cip.evals.golden_set import load_golden_set
from cip.llm.ollama_client import generate

HERE = Path(__file__).parent
GOLDEN_SET_PATH = HERE / "golden_set_labeled.csv"
FOLDS_PATH = HERE / "cv_folds.json"
LLM_RESULTS_PATH = HERE / "results_llm_classifier.json"
RESULTS_PATH = HERE / "results_cost_comparison.json"

MODEL = "llama3.2:3b"
N_TOKEN_SAMPLE = 30
RANDOM_STATE = 42

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


def time_tfidf_fold0(df, folds: dict) -> dict:
    fold_by_id = folds["fold_by_complaint_id"]
    test_ids = {cid for cid, f in fold_by_id.items() if f == 0}
    test_mask = df["complaint_id"].astype(str).isin(test_ids)
    train_df, test_df = df[~test_mask], df[test_mask]

    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2,
                                  stop_words="english")
    X_train = vectorizer.fit_transform(train_df["narrative"])
    fit_vectorizer_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train, train_df["churn_signal"])
    fit_clf_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    X_test = vectorizer.transform(test_df["narrative"])
    clf.predict(X_test)
    predict_s = time.perf_counter() - t0

    total_s = fit_vectorizer_s + fit_clf_s + predict_s
    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "fit_vectorizer_s": round(fit_vectorizer_s, 4),
        "fit_classifier_s": round(fit_clf_s, 4),
        "predict_s": round(predict_s, 4),
        "total_s": round(total_s, 4),
        "rows_per_s_predict_only": round(len(test_df) / predict_s, 1),
        "rows_per_s_full_pipeline": round((len(train_df) + len(test_df)) / total_s, 1),
    }


def llm_timing_from_existing_run() -> dict:
    results = json.loads(LLM_RESULTS_PATH.read_text())
    durations_ms = [row["eval_duration_ms"] for row in results["rows"].values()]
    arr = np.array(durations_ms, dtype=float)
    total_s = float(arr.sum()) / 1000
    return {
        "n_rows": len(durations_ms),
        "total_eval_s": round(total_s, 1),
        "total_eval_min": round(total_s / 60, 2),
        "mean_s_per_row": round(float(arr.mean()) / 1000, 3),
        "std_s_per_row": round(float(arr.std()) / 1000, 3),
        "rows_per_s": round(len(durations_ms) / total_s, 3),
    }


def sample_llm_token_counts(df) -> dict:
    sample = df.sample(n=N_TOKEN_SAMPLE, random_state=RANDOM_STATE)
    prompt_tokens, response_tokens = [], []
    for _, row in sample.iterrows():
        result = generate(
            MODEL, build_prompt(row["narrative"]), options={"temperature": 0}, format=SCHEMA
        )
        prompt_tokens.append(result.prompt_eval_count)
        response_tokens.append(result.eval_count)
    return {
        "n_sampled": N_TOKEN_SAMPLE,
        "note": "sampled estimate, NOT a full-corpus count — 04_llm_classifier.py did not "
                "record per-row token counts, only timing",
        "mean_prompt_tokens": round(float(np.mean(prompt_tokens)), 1),
        "mean_response_tokens": round(float(np.mean(response_tokens)), 1),
        "mean_total_tokens": round(float(np.mean(prompt_tokens) + np.mean(response_tokens)), 1),
    }


def main() -> int:
    df = load_golden_set(GOLDEN_SET_PATH).reset_index(drop=True)
    folds = json.loads(FOLDS_PATH.read_text())

    print("timing TF-IDF (fold 0 fit+predict)...", flush=True)
    tfidf_timing = time_tfidf_fold0(df, folds)
    print(json.dumps(tfidf_timing, indent=2))

    print("\nreading LLM timing from existing results_llm_classifier.json...", flush=True)
    llm_timing = llm_timing_from_existing_run()
    print(json.dumps(llm_timing, indent=2))

    print(f"\nsampling {N_TOKEN_SAMPLE} narratives for LLM token counts...", flush=True)
    token_estimate = sample_llm_token_counts(df)
    print(json.dumps(token_estimate, indent=2))

    speedup = round(llm_timing["rows_per_s"] and
                     tfidf_timing["rows_per_s_full_pipeline"] / llm_timing["rows_per_s"], 1)

    results = {
        "tfidf": tfidf_timing,
        "llm": {**llm_timing, "token_estimate": token_estimate},
        "tfidf_throughput_multiple_of_llm": speedup,
        "hardware_context": "15.4 GB RAM, Intel Arc iGPU (2 GB VRAM), CPU-bound for llama3.2:3b "
                             "(ADR-0001) — TF-IDF+LR runs entirely on CPU with no model weights "
                             "to load, orders of magnitude cheaper per row than an LLM call",
    }
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nTF-IDF is ~{speedup}x faster per row than the LLM (full pipeline vs LLM inference)")
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    exit(main())
