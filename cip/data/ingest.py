"""Load raw CSVs from data/raw/ into the DuckDB warehouse (data/warehouse.duckdb).

DuckDB reads the CSVs directly via read_csv_auto — no pandas round-trip needed to get
data into queryable tables.

Usage:
    uv run python -m cip.data.ingest
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
DB_PATH = REPO_ROOT / "data" / "warehouse.duckdb"

TELCO_CSV = RAW_DIR / "telco_customer_churn.csv"


def _latest_cfpb_csv() -> Path:
    candidates = sorted(RAW_DIR.glob("cfpb_complaints_*.csv"))
    if not candidates:
        sys.exit(f"no cfpb_complaints_*.csv found under {RAW_DIR}")
    return candidates[-1]


def ingest(con: duckdb.DuckDBPyConnection, cfpb_csv: Path) -> dict[str, int]:
    con.execute(f"""
        CREATE OR REPLACE TABLE telco_customers AS
        SELECT * FROM read_csv_auto('{TELCO_CSV.as_posix()}')
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE cfpb_complaints AS
        SELECT * FROM read_csv_auto('{cfpb_csv.as_posix()}', ignore_errors=true)
    """)
    return {
        "telco_customers": con.execute("SELECT count(*) FROM telco_customers").fetchone()[0],
        "cfpb_complaints": con.execute("SELECT count(*) FROM cfpb_complaints").fetchone()[0],
    }


def main() -> int:
    if not TELCO_CSV.exists():
        sys.exit(f"{TELCO_CSV} missing — run `uv run python -m cip.data.download --dataset telco`")
    cfpb_csv = _latest_cfpb_csv()

    con = duckdb.connect(str(DB_PATH))
    try:
        counts = ingest(con, cfpb_csv)
        telco_cols = con.execute("DESCRIBE telco_customers").fetchall()
        cfpb_cols = con.execute("DESCRIBE cfpb_complaints").fetchall()
    finally:
        con.close()

    print(f"warehouse: {DB_PATH}")
    print(f"telco_customers: {counts['telco_customers']:,} rows, {len(telco_cols)} columns "
          f"(source: {TELCO_CSV.name})")
    print(f"cfpb_complaints: {counts['cfpb_complaints']:,} rows, {len(cfpb_cols)} columns "
          f"(source: {cfpb_csv.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
