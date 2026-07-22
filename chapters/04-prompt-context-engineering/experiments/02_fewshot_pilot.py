"""Phase 3 pilot (~/.claude/plans/ch4-prompt-context-engineering.md): real per-row cost of the
few-shot prompt (k=4, fold-safe exemplars) before committing to a full 1,001-row run.

Each sampled row uses ONLY its own CV fold's exemplars (exemplars.py — drawn from that fold's
train partition, never the fold's own test rows), matching the leak-free out-of-fold design the
full run will use.

Run: uv run python chapters/04-prompt-context-engineering/experiments/02_fewshot_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from config import MODEL, NUM_CTX, TEMPERATURE
from exemplars import select_exemplars_per_fold
from prompts import build_prompt_fewshot

from cip.evals.golden_set import load_golden_set
from cip.llm.ollama_client import generate
from cip.llm.timing import timing_ms

HERE = Path(__file__).parent
GOLDEN_SET_PATH = HERE.parent.parent / "03-eval-harness" / "experiments" / "golden_set_labeled.csv"
FOLDS_PATH = HERE.parent.parent / "03-eval-harness" / "experiments" / "cv_folds.json"
RESULTS_PATH = HERE / "results_02_fewshot_pilot.json"

N_SAMPLE = 20
RANDOM_STATE = 43  # different draw than the calibration pilot's sample


def main() -> int:
    if not GOLDEN_SET_PATH.exists() or not FOLDS_PATH.exists():
        sys.exit("golden set or cv_folds.json missing — run Chapter 3's experiments first")

    df = load_golden_set(GOLDEN_SET_PATH)
    folds = json.loads(FOLDS_PATH.read_text())
    fold_by_id = folds["fold_by_complaint_id"]
    exemplars_by_fold = select_exemplars_per_fold(df, folds)

    sample = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE)

    rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        fold_idx = fold_by_id[str(row["complaint_id"])]
        exemplars = exemplars_by_fold[fold_idx]
        prompt, schema = build_prompt_fewshot(row["narrative"], exemplars)
        result = generate(
            MODEL, prompt, options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX}, format=schema
        )
        timing = timing_ms(result)
        rows.append({"complaint_id": str(row["complaint_id"]), "fold": fold_idx, **timing})
        print(f"  {i + 1}/{N_SAMPLE} wall={timing['wall_seconds']}s "
              f"prompt_tokens={timing['prompt_eval_count']} "
              f"prefill={timing['prompt_eval_duration_ms']}ms", flush=True)

    wall = np.array([r["wall_seconds"] for r in rows])
    prompt_tokens = np.array([r["prompt_eval_count"] for r in rows])

    def mean_std(arr: np.ndarray) -> dict:
        return {"mean": round(float(arr.mean()), 3), "std": round(float(arr.std()), 3)}

    summary = {
        "model": MODEL,
        "num_ctx": NUM_CTX,
        "n_sample": N_SAMPLE,
        "wall_seconds_per_row": mean_std(wall),
        "prompt_tokens": mean_std(prompt_tokens),
        "estimated_full_1001_rows_hours": round(float(wall.mean()) * 1001 / 3600, 2),
        "note": "1,001 total calls at full scale (one pass, fold-specific exemplars per row), "
                "NOT 1,001x5 — see plan Phase 3",
        "rows": rows,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
