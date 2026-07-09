"""Follow-up to 03_structured_output.py: does adding an `enum` constraint to a
structured-output schema change token cost — on the input side, the output side,
or both?

Two sub-experiments, both on real CFPB complaint narratives (same query shape as 03):

1. **Prompt (input) tokens** — same prompt text, called once with `format=<schema>`
   and once with `format=None`. Ollama's client sends `format` as a separate JSON
   field alongside `prompt` (`cip/llm/ollama_client.py`), not concatenated into the
   prompt string; per Ollama's XGrammar-based structured-output mechanism, the
   schema is compiled into a decode-time grammar (logit masking), not fed through
   the transformer as input text. Prediction: `prompt_eval_count` should be
   identical with and without `format`, enum or no enum, since the schema never
   enters the token stream. Uses `num_predict=1` (see `tokenize_via_eval_count`)
   so we pay for prompt processing only, not a full generation.

2. **Completion (output) tokens** — same prompt, same schema shape, differing only
   in whether `product` is a free `{"type": "string"}` or constrained to the 11 real
   CFPB `Product` categories via `enum`. Prediction: constraining a field to a closed
   set of (mostly short) known strings should not increase output length, and may
   decrease it if the unconstrained model tends to produce longer/hedged phrasing.
   This is the one that isn't guaranteed by the decoding mechanism the way (1) is —
   it depends on what the model would otherwise have generated.

Run: uv run python chapters/02-llm-mechanics/experiments/05_schema_token_cost.py
"""

import json
from pathlib import Path

import duckdb

from cip.llm.ollama_client import generate

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"
OUT_PATH = Path(__file__).parent / "results_schema_token_cost.json"

MODEL = "llama3.2:3b"
N_SAMPLES = 10

CFPB_PRODUCTS = [
    "Checking or savings account",
    "Credit card",
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Debt or credit management",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Payday loan, title loan, personal loan, or advance loan",
    "Prepaid card",
    "Student loan",
    "Vehicle loan or lease",
]

SCHEMA_NO_ENUM = {
    "type": "object",
    "properties": {
        "product": {"type": "string"},
        "issue_summary": {"type": "string"},
        "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
        "mentions_legal_action": {"type": "boolean"},
    },
    "required": ["product", "issue_summary", "urgency", "mentions_legal_action"],
}

SCHEMA_ENUM = {
    "type": "object",
    "properties": {
        "product": {"type": "string", "enum": CFPB_PRODUCTS},
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


def build_prompt(text: str) -> str:
    return (
        "Extract structured information from this consumer complaint narrative. "
        f"Return JSON matching the schema.\n\n{text}"
    )


def main() -> None:
    samples = load_samples(N_SAMPLES)

    prompt_rows = []
    for sample in samples:
        prompt = build_prompt(sample["text"])
        no_format = generate(MODEL, prompt, options={"temperature": 0, "num_predict": 1})
        with_format = generate(
            MODEL, prompt, options={"temperature": 0, "num_predict": 1}, format=SCHEMA_NO_ENUM
        )
        prompt_rows.append({
            "complaint_id": sample["complaint_id"],
            "prompt_eval_count_no_format": no_format.prompt_eval_count,
            "prompt_eval_count_with_format": with_format.prompt_eval_count,
            "delta": with_format.prompt_eval_count - no_format.prompt_eval_count,
        })
        print(f"[prompt-cost] {sample['complaint_id']}: "
              f"no_format={no_format.prompt_eval_count} "
              f"with_format={with_format.prompt_eval_count}")

    completion_rows = []
    for sample in samples:
        prompt = build_prompt(sample["text"])
        no_enum = generate(MODEL, prompt, options={"temperature": 0}, format=SCHEMA_NO_ENUM)
        with_enum = generate(MODEL, prompt, options={"temperature": 0}, format=SCHEMA_ENUM)

        def safe_parse(resp: str) -> dict | None:
            try:
                return json.loads(resp)
            except json.JSONDecodeError:
                return None

        no_enum_obj = safe_parse(no_enum.response)
        with_enum_obj = safe_parse(with_enum.response)
        completion_rows.append({
            "complaint_id": sample["complaint_id"],
            "true_product": sample["true_product"],
            "eval_count_no_enum": no_enum.eval_count,
            "eval_count_with_enum": with_enum.eval_count,
            "delta": with_enum.eval_count - no_enum.eval_count,
            "product_no_enum": no_enum_obj.get("product") if no_enum_obj else None,
            "product_with_enum": with_enum_obj.get("product") if with_enum_obj else None,
        })
        print(f"[completion-cost] {sample['complaint_id']}: "
              f"no_enum={no_enum.eval_count}tok product={no_enum_obj.get('product') if no_enum_obj else None!r} | "
              f"with_enum={with_enum.eval_count}tok product={with_enum_obj.get('product') if with_enum_obj else None!r}")

    prompt_deltas = [r["delta"] for r in prompt_rows]
    completion_deltas = [r["delta"] for r in completion_rows]

    summary = {
        "model": MODEL,
        "n_samples": N_SAMPLES,
        "prompt_token_cost": {
            "mean_no_format": round(sum(r["prompt_eval_count_no_format"] for r in prompt_rows) / N_SAMPLES, 1),
            "mean_with_format": round(sum(r["prompt_eval_count_with_format"] for r in prompt_rows) / N_SAMPLES, 1),
            "mean_delta": round(sum(prompt_deltas) / N_SAMPLES, 2),
            "max_abs_delta": max(abs(d) for d in prompt_deltas),
            "rows": prompt_rows,
        },
        "completion_token_cost": {
            "mean_no_enum": round(sum(r["eval_count_no_enum"] for r in completion_rows) / N_SAMPLES, 1),
            "mean_with_enum": round(sum(r["eval_count_with_enum"] for r in completion_rows) / N_SAMPLES, 1),
            "mean_delta": round(sum(completion_deltas) / N_SAMPLES, 2),
            "rows": completion_rows,
        },
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print("\n--- summary ---")
    print(f"prompt tokens: no_format mean={summary['prompt_token_cost']['mean_no_format']} "
          f"with_format mean={summary['prompt_token_cost']['mean_with_format']} "
          f"(max |delta|={summary['prompt_token_cost']['max_abs_delta']})")
    print(f"completion tokens: no_enum mean={summary['completion_token_cost']['mean_no_enum']} "
          f"with_enum mean={summary['completion_token_cost']['mean_with_enum']} "
          f"(mean delta={summary['completion_token_cost']['mean_delta']})")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
