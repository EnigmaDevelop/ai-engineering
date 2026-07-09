"""Download the portfolio's datasets into data/raw/.

Sources (both real, public):
- Telco: IBM's official sample repo (github.com/IBM/telco-customer-churn-on-icp4d).
- CFPB:  Consumer Complaint Database. Default pulls a date-filtered CSV through the public
  search API; --full fetches the complete ~gigabyte-scale archive zip instead.

Usage:
    uv run python -m cip.data.download --dataset telco
    uv run python -m cip.data.download --dataset cfpb --since 2026-01-01
    uv run python -m cip.data.download --dataset cfpb --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

TELCO_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)
CFPB_API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
CFPB_FULL_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"

CHUNK = 1 << 20  # 1 MiB


def _stream_to(url: str, dest: Path, params: dict | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, params=params, stream=True, timeout=(10, 300)) as resp:
        resp.raise_for_status()
        size = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                fh.write(chunk)
                size += len(chunk)
                print(f"\r{dest.name}: {size / 1e6:,.1f} MB", end="", flush=True)
    print()
    return dest


def download_telco() -> Path:
    return _stream_to(TELCO_URL, RAW_DIR / "telco_customer_churn.csv")


def download_cfpb(since: str | None, until: str | None, full: bool) -> Path:
    if full:
        return _stream_to(CFPB_FULL_URL, RAW_DIR / "cfpb_complaints_full.csv.zip")
    params: dict[str, str] = {
        "format": "csv",
        "field": "all",
        "no_aggs": "true",
        "size": "0",
    }
    if since:
        params["date_received_min"] = since
    if until:
        params["date_received_max"] = until
    suffix = f"{since or 'start'}_{until or 'now'}".replace("-", "")
    return _stream_to(CFPB_API_URL, RAW_DIR / f"cfpb_complaints_{suffix}.csv", params=params)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["telco", "cfpb"], required=True)
    parser.add_argument("--since", help="CFPB: date_received_min, YYYY-MM-DD")
    parser.add_argument("--until", help="CFPB: date_received_max, YYYY-MM-DD")
    parser.add_argument("--full", action="store_true", help="CFPB: full archive zip")
    args = parser.parse_args(argv)

    if args.dataset == "telco":
        dest = download_telco()
    else:
        dest = download_cfpb(args.since, args.until, args.full)
    print(f"saved: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
