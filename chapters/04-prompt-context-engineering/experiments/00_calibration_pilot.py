"""Phase 0.5 calibration pilot (~/.claude/plans/ch4-prompt-context-engineering.md).

Re-runs condition A (Chapter 3's zero-shot prompt, byte-identical wording — see prompts.py's
docstring) on a fresh sample, logging ALL of Ollama's timing fields (cip/llm/timing.py) instead of
just decode duration. Chapter 3's own "3.0s/row" figure was decode-only and never included prefill,
which is why it looked inconsistent with the ~16.6ms/token prefill rate measured separately during
Ch.4 planning (a 4,098-token prompt took 68.06s of prefill alone). This pilot's real per-row
wall-clock number — not a bracket guess — is what every later Ch.4 phase's time estimate gets
rescaled against.

Uses config.NUM_CTX (the same value every other Ch.4 experiment will use) so this calibration is
representative of the real runs, not a cheaper stand-in.

Run: uv run python chapters/04-prompt-context-engineering/experiments/00_calibration_pilot.py
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
RESULTS_PATH = HERE / "results_00_calibration_pilot.json"

N_SAMPLE = 25
RANDOM_STATE = 42


def main() -> int:
    if not GOLDEN_SET_PATH.exists():
        sys.exit(f"{GOLDEN_SET_PATH} missing")

    df = load_golden_set(GOLDEN_SET_PATH)
    sample = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE)

    rows = []
    for i, (_, row) in enumerate(sample.iterrows()):
        prompt, schema = build_prompt_cot(row["narrative"], "A")
        result = generate(
            MODEL, prompt, options={"temperature": TEMPERATURE, "num_ctx": NUM_CTX}, format=schema
        )
        timing = timing_ms(result)
        rows.append({"complaint_id": str(row["complaint_id"]), **timing})
        print(f"  {i + 1}/{N_SAMPLE} wall={timing['wall_seconds']}s "
              f"prefill={timing['prompt_eval_duration_ms']}ms "
              f"decode={timing['eval_duration_ms']}ms", flush=True)

    wall = np.array([r["wall_seconds"] for r in rows])
    prefill_ms = np.array([r["prompt_eval_duration_ms"] for r in rows])
    decode_ms = np.array([r["eval_duration_ms"] for r in rows])
    prompt_tokens = np.array([r["prompt_eval_count"] for r in rows])

    def mean_std(arr: np.ndarray) -> dict:
        return {"mean": round(float(arr.mean()), 3), "std": round(float(arr.std()), 3)}

    summary = {
        "model": MODEL,
        "num_ctx": NUM_CTX,
        "n_sample": N_SAMPLE,
        "wall_seconds_per_row": mean_std(wall),
        "prompt_eval_duration_ms": mean_std(prefill_ms),
        "eval_duration_ms": mean_std(decode_ms),
        "prompt_tokens": mean_std(prompt_tokens),
        "estimated_full_1001_rows_hours": round(float(wall.mean()) * 1001 / 3600, 2),
        "estimated_full_1001_rows_hours_vs_ch3_claim": (
            "Ch.3 reported 50 min (0.83 hr) total, decode-only — compare against this "
            "wall-clock-inclusive estimate to see how much prefill was missing from that figure"
        ),
        "rows": rows,
    }
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
