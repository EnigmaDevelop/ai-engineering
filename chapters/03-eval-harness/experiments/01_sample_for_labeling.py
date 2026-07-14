"""Stratified sample of real CFPB complaint narratives for golden-set labeling.

CFPB has no native churn label (unlike Telco). This script draws a reproducible sample across
the 3 CFPB products with both the most real narrative text and the clearest "ongoing customer
relationship" framing (measured against data/warehouse.duckdb): Debt collection, Checking or
savings account, Credit card.

Sampling is deterministic via hash() ordering (not random()+setseed, which is not guaranteed
stable across DuckDB's multi-threaded execution) — rerunning this script reproduces the exact
same rows.

The label is deliberately named churn_signal, not churn_risk: nobody can know from CFPB data
whether the consumer actually churned. What CAN be labeled from the text — by a non-expert,
reliably — is whether the narrative contains an EXPLICIT statement of leaving:

    churn_signal = 1  the narrative explicitly states the consumer closed / is closing the
                      account, switched / is switching provider, or states a definite intention
                      or threat to leave because of the issue described.
    churn_signal = 0  otherwise — complaint, dispute, anger, but no statement about ending
                      the relationship.

This is text-evidence detection, not churn prediction. Labeling pipeline (hybrid, human-final):
02_prelabel_claude.py proposes a label + verbatim evidence quote per narrative via `claude -p`
(subscription auth, zero extra spend, different model family from the local models under
evaluation); label_cli.py then walks the annotator through every proposal to accept, override,
or mark ambiguous. The human-verified file is committed as golden_set_labeled.csv.

Run: uv run python chapters/03-eval-harness/experiments/01_sample_for_labeling.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"
OUT_PATH = Path(__file__).parent / "golden_set_template.csv"

MIN_LEN = 200
MAX_LEN = 2000

# Per-product sample size — deterministic hash ordering means growing a product's N only
# EXTENDS its existing rows, never replaces them.
#
# Debt collection is capped at 167 (not grown further): after 167/167 human-reviewed rows
# produced zero churn_signal=1, and a structural reason why (see docs/study-notes/09 —
# a debt collector is an assigned relationship the consumer never chose, so "churn" from one
# is close to incoherent), further sampling here has near-zero expected return. Labeling
# effort is concentrated on the two products with a real voluntary customer relationship.
N_PER_PRODUCT = {
    "Debt collection": 167,
    "Checking or savings account": 417,
    "Credit card": 417,
}
PRODUCTS = list(N_PER_PRODUCT)


def sample_product(con: duckdb.DuckDBPyConnection, product: str, n: int) -> list[tuple]:
    return con.execute(
        """
        select "Complaint ID", "Product", "Consumer complaint narrative"
        from cfpb_complaints
        where "Product" = ?
          and "Consumer complaint narrative" is not null
          and length("Consumer complaint narrative") between ? and ?
        order by hash(cast("Complaint ID" as varchar))
        limit ?
        """,
        [product, MIN_LEN, MAX_LEN, n],
    ).fetchall()


def main() -> int:
    if not DB_PATH.exists():
        sys.exit(f"{DB_PATH} missing — run `uv run python -m cip.data.ingest` first (Ch.0)")

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = []
        for product in PRODUCTS:
            product_rows = sample_product(con, product, N_PER_PRODUCT[product])
            print(f"{product}: sampled {len(product_rows)} narratives")
            rows.extend(product_rows)
    finally:
        con.close()

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["complaint_id", "product", "narrative", "churn_signal"])
        for complaint_id, product, narrative in rows:
            writer.writerow([complaint_id, product, narrative, ""])

    print(f"\nwrote {len(rows)} rows to {OUT_PATH}")
    print("next: uv run python chapters/03-eval-harness/experiments/02_prelabel_claude.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
