"""Measures the effect of temperature and top_p on output determinism/diversity,
on a real task: one-sentence summarization of a real CFPB complaint narrative.

Method: fix the prompt, repeat generation N times per setting, measure:
- distinct_outputs: how many of the N completions are byte-identical
- avg_pairwise_jaccard: average word-set Jaccard similarity across all pairs of
  completions (1.0 = every completion used exactly the same words, 0.0 = no
  word overlap at all) — a diversity signal beyond exact-match

Run: uv run python chapters/02-llm-mechanics/experiments/02_sampling.py
"""

import itertools
import json
from pathlib import Path

import duckdb

from cip.llm.ollama_client import generate

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"
OUT_PATH = Path(__file__).parent / "results_sampling.json"

MODEL = "llama3.2:3b"
N_REPEATS = 8
TEMPERATURES = [0.0, 0.3, 0.7, 1.0, 1.5]
TOP_PS = [0.1, 0.5, 0.9, 1.0]
FIXED_TEMPERATURE_FOR_TOP_P_SWEEP = 0.8


def load_one_narrative() -> str:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    text = con.execute(
        """
        select "Consumer complaint narrative"
        from cfpb_complaints
        where "Consumer complaint narrative" is not null
          and length("Consumer complaint narrative") between 300 and 500
        order by "Complaint ID"
        limit 1
        """
    ).fetchone()[0]
    con.close()
    return text


def jaccard(a: str, b: str) -> float:
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def diversity_stats(outputs: list[str]) -> dict:
    distinct = len(set(outputs))
    pairs = list(itertools.combinations(outputs, 2))
    avg_jaccard = sum(jaccard(a, b) for a, b in pairs) / len(pairs) if pairs else 1.0
    return {
        "distinct_outputs": distinct,
        "n": len(outputs),
        "avg_pairwise_jaccard": round(avg_jaccard, 4),
    }


def run_sweep(prompt: str, param_name: str, values: list[float], fixed: dict) -> dict:
    sweep_results = {}
    for value in values:
        options = {**fixed, param_name: value}
        outputs = [generate(MODEL, prompt, options=options).response.strip() for _ in range(N_REPEATS)]
        stats = diversity_stats(outputs)
        sweep_results[str(value)] = {**stats, "outputs": outputs}
        print(f"{param_name}={value}: distinct={stats['distinct_outputs']}/{stats['n']} "
              f"avg_jaccard={stats['avg_pairwise_jaccard']}")
    return sweep_results


def main() -> None:
    narrative = load_one_narrative()
    prompt = f"In exactly one sentence, summarize this customer complaint:\n\n{narrative}"

    print("=== Temperature sweep (top_p left at Ollama default) ===")
    temp_results = run_sweep(prompt, "temperature", TEMPERATURES, fixed={})

    print("\n=== top_p sweep (temperature fixed at 0.8) ===")
    top_p_results = run_sweep(
        prompt, "top_p", TOP_PS, fixed={"temperature": FIXED_TEMPERATURE_FOR_TOP_P_SWEEP}
    )

    results = {
        "model": MODEL,
        "n_repeats": N_REPEATS,
        "prompt_narrative_char_len": len(narrative),
        "temperature_sweep": temp_results,
        "top_p_sweep": top_p_results,
    }
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
