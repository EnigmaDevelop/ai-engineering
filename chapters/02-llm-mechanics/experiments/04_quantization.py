"""Compares llama3.2:3b at multiple quantization levels on three axes: disk size,
latency (cold load + steady-state tokens/sec), and a quality proxy — reusing the
same structured-extraction task and loose_product_match_rate signal from
03_structured_output.py, now applied across quant levels instead of across models.

Cold load is measured by unloading the model (keep_alive=0) before each model's
first call, so load_duration_ns reflects an actual cold read from disk, not a
warm no-op.

Run: uv run python chapters/02-llm-mechanics/experiments/04_quantization.py
"""

import json
from pathlib import Path

import duckdb
import requests

from cip.llm.ollama_client import BASE_URL, generate, list_models

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"
OUT_PATH = Path(__file__).parent / "results_quantization.json"

MODELS = ["llama3.2:3b", "llama3.2:3b-instruct-q8_0", "llama3.2:3b-instruct-fp16"]
N_SAMPLES = 8

SCHEMA = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "issue_summary": {"type": "string"},
        "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
        "mentions_legal_action": {"type": "boolean"},
    },
    "required": ["product", "issue_summary", "urgency", "mentions_legal_action"],
}


def load_samples(n: int) -> list[dict]:
    con = duckdb.connect(str(DB_PATH), read_only=True)
    rows = con.execute(
        """
        select "Complaint ID", "Product", "Consumer complaint narrative"
        from cfpb_complaints
        where "Consumer complaint narrative" is not null
          and length("Consumer complaint narrative") between 300 and 600
        order by "Complaint ID"
        limit ?
        """,
        [n],
    ).fetchall()
    con.close()
    return [{"complaint_id": r[0], "true_product": r[1], "text": r[2]} for r in rows]


def loose_product_match(true_product: str, extracted_product: str) -> bool:
    return true_product.split()[0].lower() in extracted_product.lower()


def unload(model: str) -> None:
    requests.post(f"{BASE_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)


def disk_sizes() -> dict[str, int]:
    return {m["model"]: m["size"] for m in list_models()}


def main() -> None:
    samples = load_samples(N_SAMPLES)
    sizes = disk_sizes()
    results = {}

    for model in MODELS:
        if model not in sizes:
            print(f"skipping {model}: not pulled locally")
            continue

        unload(model)
        cold = generate(model, "Say OK.", options={"temperature": 0, "num_predict": 5})
        warm = generate(model, "Say OK.", options={"temperature": 0, "num_predict": 5})

        n_schema_valid = 0
        n_loose_match = 0
        eval_ms = []
        tokens_per_sec = []
        for sample in samples:
            prompt = (
                "Extract structured information from this consumer complaint narrative. "
                f"Return JSON matching the schema.\n\n{sample['text']}"
            )
            result = generate(model, prompt, options={"temperature": 0}, format=SCHEMA)
            eval_ms.append(result.eval_duration_ns / 1e6)
            if result.eval_duration_ns > 0:
                tokens_per_sec.append(result.eval_count / (result.eval_duration_ns / 1e9))
            try:
                extracted = json.loads(result.response)
                required_ok = all(k in extracted for k in SCHEMA["required"])
                if required_ok:
                    n_schema_valid += 1
                    if loose_product_match(sample["true_product"], extracted["product"]):
                        n_loose_match += 1
            except json.JSONDecodeError:
                pass

        results[model] = {
            "disk_size_bytes": sizes[model],
            "disk_size_gb": round(sizes[model] / 1e9, 2),
            "cold_load_duration_ms": round(cold.load_duration_ns / 1e6, 1),
            "warm_load_duration_ms": round(warm.load_duration_ns / 1e6, 1),
            "mean_tokens_per_sec": round(sum(tokens_per_sec) / len(tokens_per_sec), 1)
            if tokens_per_sec else None,
            "mean_eval_duration_ms": round(sum(eval_ms) / len(eval_ms), 1),
            "schema_valid_rate": round(n_schema_valid / N_SAMPLES, 3),
            "loose_product_match_rate": round(n_loose_match / N_SAMPLES, 3),
        }
        r = results[model]
        print(f"{model}: {r['disk_size_gb']}GB cold_load={r['cold_load_duration_ms']}ms "
              f"tok/s={r['mean_tokens_per_sec']} schema_valid={r['schema_valid_rate']} "
              f"loose_match={r['loose_product_match_rate']}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
