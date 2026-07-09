"""Measures tokenizer behavior on real CFPB complaint narratives across two local
models with different tokenizer families (Llama's vs Qwen's BPE vocab).

Two things are measured, not assumed:
1. Tokens-per-character for the same real text, across models — different
   tokenizers segment the same string differently.
2. The chat-template "tax": prompt_eval_count with raw=True (no template) vs
   raw=False (model's default instruct/system template applied) — the token
   cost of the template itself, before any user content.

Run: uv run python chapters/02-llm-mechanics/experiments/01_tokenization.py
"""

import json
from pathlib import Path

import duckdb

from cip.llm.ollama_client import tokenize_via_eval_count

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"
OUT_PATH = Path(__file__).parent / "results_tokenization.json"

MODELS = ["llama3.2:3b", "qwen2.5:3b"]


def load_sample_narratives(n: int = 5) -> list[dict]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        """
        select "Complaint ID", "Consumer complaint narrative"
        from cfpb_complaints
        where "Consumer complaint narrative" is not null
          and length("Consumer complaint narrative") between 200 and 600
        order by "Complaint ID"
        limit ?
        """,
        [n],
    ).fetchall()
    con.close()
    return [{"complaint_id": cid, "text": text} for cid, text in rows]


def main() -> None:
    samples = load_sample_narratives(5)
    results = {"samples": [], "template_tax": {}}

    for sample in samples:
        text = sample["text"]
        row = {"complaint_id": sample["complaint_id"], "char_len": len(text), "models": {}}
        for model in MODELS:
            n_tokens = tokenize_via_eval_count(model, text, raw=True)
            row["models"][model] = {
                "raw_token_count": n_tokens,
                "chars_per_token": round(len(text) / n_tokens, 3),
            }
        results["samples"].append(row)
        print(f"complaint {sample['complaint_id']} ({len(text)} chars): "
              + ", ".join(f"{m}={row['models'][m]['raw_token_count']}tok" for m in MODELS))

    # Chat-template tax: one representative short prompt, raw vs templated.
    probe = "Hello, how are you today?"
    for model in MODELS:
        raw_count = tokenize_via_eval_count(model, probe, raw=True)
        templated_count = tokenize_via_eval_count(model, probe, raw=False)
        results["template_tax"][model] = {
            "probe": probe,
            "raw_tokens": raw_count,
            "templated_tokens": templated_count,
            "template_overhead_tokens": templated_count - raw_count,
        }
        print(f"{model} template tax: raw={raw_count} templated={templated_count} "
              f"overhead={templated_count - raw_count}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
