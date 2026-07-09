"""Extends Ch.0's single structured-output smoke test into a measured experiment:
schema-conformance rate, latency, and a loose sanity check against CFPB's own
ground-truth Product label, across N real complaint narratives.

Scope note: this is NOT a rigorous accuracy eval (that's Ch.3's job, with a real
golden set and LLM-as-judge methodology). The "loose_product_match" field here is
an honest, cheap sanity signal only — a substring check against CFPB's own Product
label — not a scored metric to be quoted as "accuracy."

Run: uv run python chapters/02-llm-mechanics/experiments/03_structured_output.py
"""

import json
from pathlib import Path

import duckdb

from cip.llm.ollama_client import generate

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"
OUT_PATH = Path(__file__).parent / "results_structured_output.json"

MODEL = "llama3.2:3b"
N_SAMPLES = 15

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


def validate_schema(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    required = SCHEMA["required"]
    if not all(k in obj for k in required):
        return False
    if not isinstance(obj["product"], str) or not isinstance(obj["issue_summary"], str):
        return False
    if obj["urgency"] not in ("low", "medium", "high"):
        return False
    if not isinstance(obj["mentions_legal_action"], bool):
        return False
    return True


def loose_product_match(true_product: str, extracted_product: str) -> bool:
    true_first_word = true_product.split()[0].lower()
    return true_first_word in extracted_product.lower()


def main() -> None:
    samples = load_samples(N_SAMPLES)
    rows = []
    n_parsed = 0
    n_schema_valid = 0
    n_loose_match = 0
    eval_durations_ms = []

    for sample in samples:
        prompt = (
            "Extract structured information from this consumer complaint narrative. "
            f"Return JSON matching the schema.\n\n{sample['text']}"
        )
        result = generate(MODEL, prompt, options={"temperature": 0}, format=SCHEMA)
        eval_durations_ms.append(result.eval_duration_ns / 1e6)

        parsed_ok = False
        schema_ok = False
        extracted = None
        try:
            extracted = json.loads(result.response)
            parsed_ok = True
            schema_ok = validate_schema(extracted)
        except json.JSONDecodeError:
            pass

        n_parsed += int(parsed_ok)
        n_schema_valid += int(schema_ok)
        match = False
        if schema_ok:
            match = loose_product_match(sample["true_product"], extracted["product"])
            n_loose_match += int(match)

        rows.append({
            "complaint_id": sample["complaint_id"],
            "true_product": sample["true_product"],
            "json_parsed": parsed_ok,
            "schema_valid": schema_ok,
            "extracted": extracted,
            "loose_product_match": match,
            "eval_duration_ms": round(result.eval_duration_ns / 1e6, 1),
        })
        print(f"{sample['complaint_id']}: parsed={parsed_ok} schema_valid={schema_ok} "
              f"loose_match={match} ({round(result.eval_duration_ns / 1e6)}ms)")

    summary = {
        "model": MODEL,
        "n_samples": N_SAMPLES,
        "json_parse_rate": round(n_parsed / N_SAMPLES, 3),
        "schema_valid_rate": round(n_schema_valid / N_SAMPLES, 3),
        "loose_product_match_rate": round(n_loose_match / N_SAMPLES, 3),
        "eval_duration_ms_mean": round(sum(eval_durations_ms) / len(eval_durations_ms), 1),
        "eval_duration_ms_min": round(min(eval_durations_ms), 1),
        "eval_duration_ms_max": round(max(eval_durations_ms), 1),
        "rows": rows,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\njson_parse_rate={summary['json_parse_rate']} "
          f"schema_valid_rate={summary['schema_valid_rate']} "
          f"loose_product_match_rate={summary['loose_product_match_rate']}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
