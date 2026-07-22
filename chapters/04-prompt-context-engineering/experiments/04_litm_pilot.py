"""Phase 4 pilot: real per-call cost at the ~3,000-token confirmatory Lost-in-the-Middle length,
before committing to the full 96-target x 3-position (288-call) run.

Filler pool: real Debt collection narratives (churn_signal=0 for all 167, structural per Ch.3's
finding — see chapters/03-eval-harness/README.md). A handful of real targets (mixed pos/neg), each
tested at all 3 positions with the SAME filler draw (paired design — only the target moves).

Run: uv run python chapters/04-prompt-context-engineering/experiments/04_litm_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from config import MODEL, NUM_CTX, TEMPERATURE
from litm_documents import assemble_context, build_filler_pool
from prompts import build_prompt_litm

from cip.evals.golden_set import load_golden_set
from cip.llm.ollama_client import generate
from cip.llm.timing import timing_ms

HERE = Path(__file__).parent
GOLDEN_SET_PATH = HERE.parent.parent / "03-eval-harness" / "experiments" / "golden_set_labeled.csv"
RESULTS_PATH = HERE / "results_04_litm_pilot.json"

N_TARGETS = 4  # 2 positive + 2 negative
RANDOM_STATE = 45
CONTEXT_TOKEN_BUDGET = 3000
POSITIONS = ["start", "middle", "end"]


def main() -> int:
    if not GOLDEN_SET_PATH.exists():
        sys.exit(f"{GOLDEN_SET_PATH} missing")

    df = load_golden_set(GOLDEN_SET_PATH)
    filler_pool = build_filler_pool(df, product="Debt collection")

    positives = df[df["churn_signal"] == 1].sample(n=N_TARGETS // 2, random_state=RANDOM_STATE)
    negatives = df[df["churn_signal"] == 0].sample(n=N_TARGETS // 2, random_state=RANDOM_STATE)
    targets = list(positives.itertuples()) + list(negatives.itertuples())

    rows = []
    call_index = 0
    for target in targets:
        target_id = f"target_{target.complaint_id}"
        for position in POSITIONS:
            call_index += 1
            documents = assemble_context(
                target_id, target.narrative, filler_pool, position,
                CONTEXT_TOKEN_BUDGET, seed=int(target.complaint_id),
            )
            prompt, schema = build_prompt_litm(documents, target_id)
            result = generate(
                MODEL, prompt, options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
                format=schema,
            )
            timing = timing_ms(result)
            rows.append({
                "target_complaint_id": str(target.complaint_id),
                "true_label": int(target.churn_signal),
                "position": position,
                "n_documents": len(documents),
                **timing,
            })
            print(f"  {call_index}/{N_TARGETS * len(POSITIONS)} target={target.complaint_id} "
                  f"pos={position} wall={timing['wall_seconds']}s "
                  f"prompt_tokens={timing['prompt_eval_count']}", flush=True)

    wall = np.array([r["wall_seconds"] for r in rows])
    prompt_tokens = np.array([r["prompt_eval_count"] for r in rows])

    def mean_std(arr: np.ndarray) -> dict:
        return {"mean": round(float(arr.mean()), 3), "std": round(float(arr.std()), 3)}

    summary = {
        "model": MODEL,
        "num_ctx": NUM_CTX,
        "context_token_budget": CONTEXT_TOKEN_BUDGET,
        "n_targets": N_TARGETS,
        "n_calls": len(rows),
        "wall_seconds_per_call": mean_std(wall),
        "prompt_tokens": mean_std(prompt_tokens),
        "estimated_full_288_calls_hours": round(float(wall.mean()) * 288 / 3600, 2),
        "rows": rows,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
