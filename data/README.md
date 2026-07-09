# data/

- `raw/` — downloaded datasets, **gitignored** (large; re-fetch with
  `uv run python -m cip.data.download`).
- Small samples needed by tests are committed explicitly outside `raw/`.
- `warehouse.duckdb` is a build artifact, gitignored; rebuilt from raw + ingestion code.
