"""Phase 2 pilot: real per-row cost for the three new CoT conditions (B, C, D — condition A is
Ch.3's zero-shot, already measured, not rerun). Calibrates whether the reasoning-first schema
(B, D) meaningfully inflates response token count / decode time, not just prompt structure.

Run: uv run python chapters/04-prompt-context-engineering/experiments/03_cot_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from config import MODEL, NUM_CTX, TEMPERATURE
from prompts import build_prompt_cot

from cip.evals.golden_set import load_golden_set
from cip.llm.ollama_client import generate
from cip.llm.timing import timing_ms

HERE = Path(__file__).parent
GOLDEN_SET_PATH = HERE.parent.parent / "03-eval-harness" / "experiments" / "golden_set_labeled.csv"
RESULTS_PATH = HERE / "results_03_cot_pilot.json"

N_SAMPLE = 20
RANDOM_STATE = 44
CONDITIONS = ["B", "C", "D"]


def main() -> int:
    if not GOLDEN_SET_PATH.exists():
        sys.exit(f"{GOLDEN_SET_PATH} missing")

    df = load_golden_set(GOLDEN_SET_PATH)
    sample = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE)

    results_by_condition = {}
    for condition in CONDITIONS:
        rows = []
        for i, (_, row) in enumerate(sample.iterrows()):
            prompt, schema = build_prompt_cot(row["narrative"], condition)
            result = generate(
                MODEL, prompt, options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
                format=schema,
            )
            timing = timing_ms(result)
            rows.append({"complaint_id": str(row["complaint_id"]), **timing})
            print(f"  [{condition}] {i + 1}/{N_SAMPLE} wall={timing['wall_seconds']}s "
                  f"eval_count={timing['eval_count']}", flush=True)

        wall = np.array([r["wall_seconds"] for r in rows])
        eval_count = np.array([r["eval_count"] for r in rows])

        def mean_std(arr: np.ndarray) -> dict:
            return {"mean": round(float(arr.mean()), 3), "std": round(float(arr.std()), 3)}

        results_by_condition[condition] = {
            "wall_seconds_per_row": mean_std(wall),
            "response_tokens": mean_std(eval_count),
            "estimated_full_1001_rows_hours": round(float(wall.mean()) * 1001 / 3600, 2),
            "rows": rows,
        }

    summary = {"model": MODEL, "num_ctx": NUM_CTX, "n_sample": N_SAMPLE,
               "conditions": results_by_condition}
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(
        {c: {k: v for k, v in r.items() if k != "rows"} for c, r in results_by_condition.items()},
        indent=2,
    ))
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
