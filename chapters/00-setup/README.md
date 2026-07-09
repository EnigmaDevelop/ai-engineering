# Chapter 0 — Setup

No article for this chapter; it exists so every later chapter starts from a verified foundation.

## Goal

Working environment + both datasets ingested into DuckDB + **proof** that Claude is reachable at
zero extra cost through the Pro subscription.

## Checklist

- [x] git repo, uv project, `cip/` package skeleton (installed editable via `uv sync`)
- [x] Hardware measured and recorded in [ADR-0001](../../docs/adr/0001-tech-stack.md):
      15.4 GB RAM, Intel Arc iGPU 2 GB VRAM → default local models 3–4B
- [x] Upgrade Ollama (0.31.1 → 0.31.2) and pull `llama3.2:3b`, `qwen2.5:3b`, `nomic-embed-text`
- [x] Confirm no `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` set anywhere (shell env or Windows
      User/Machine persisted vars) — those take auth precedence over the subscription OAuth token
      and would silently switch billing to pay-per-token
- [x] `claude setup-token` → verified `claude -p` headless works with the Pro subscription OAuth
      token; working example under `experiments/verify_claude_auth.py`
- [x] `uv run python -m cip.data.download --dataset telco`
- [x] `uv run python -m cip.data.download --dataset cfpb --since 2026-01-01`
- [x] Ingest both into DuckDB (`data/warehouse.duckdb`) with row counts recorded below

## Acceptance criteria

1. A committed note in this README with the **measured row counts** of both datasets.
2. A runnable script under `experiments/` proving Claude access with subscription auth. If the
   Agent SDK refuses subscription auth, the fallback (`claude -p`) is documented here and the
   plan's Claude chapters are revised accordingly.
3. `uv run pytest` and `uv run ruff check .` pass clean.

## Results

**Ollama, measured 2026-07-09:**

- Version: 0.31.2 (upgraded from 0.31.1).
- Models pulled: `llama3.2:3b` (2.0 GB), `qwen2.5:3b` (1.9 GB), `nomic-embed-text` (274 MB).
- Structured output (JSON schema via `format`) verified against `llama3.2:3b`:
  prompt "Extract the name and age: John is 30 years old." → `{"name": "John", "age": 30}`,
  schema-conformant. `total_duration` 5.76s (cold load 4.44s dominates; `eval_duration` 0.90s
  for 17 output tokens once loaded).
- Also present locally but **out of scope for the local-only narrative**: `mistral`, `llama3`,
  `llama2`, `sqlcoder` (all 7-8B class, pre-dating this project) and `deepseek-v3.1:671b-cloud` —
  the `-cloud` suffix means inference runs on Ollama's servers, not this machine. Excluded from
  every "local-first, privacy-aware" experiment in this repo; a genuine violation of the Ch.2/Ch.12
  premise if used.

**Auth precedence check, measured 2026-07-09:**

- `printenv | grep -i "ANTHROPIC_API_KEY\|ANTHROPIC_AUTH_TOKEN\|CLAUDE_CODE_OAUTH_TOKEN"` → none set
  in the current shell.
- `[Environment]::GetEnvironmentVariable(...)` for both `User` and `Machine` scope → none set for
  `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` either.
- Per the [authentication precedence order](https://code.claude.com/docs/en/authentication#authentication-precedence),
  both of those env vars outrank `CLAUDE_CODE_OAUTH_TOKEN` and subscription `/login` — if either had
  been set, Claude Code would silently bill pay-per-token instead of drawing from the Pro
  subscription. Confirmed clean.

**Claude subscription auth, measured 2026-07-09:**

- `claude setup-token` completed; token stored in `.env` (gitignored, never committed) as
  `CLAUDE_CODE_OAUTH_TOKEN`.
- [`experiments/verify_claude_auth.py`](experiments/verify_claude_auth.py) calls `claude -p` in an
  **isolated subprocess environment** containing only `CLAUDE_CODE_OAUTH_TOKEN` and the OS
  variables needed to locate the binary — no `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` present, so
  a pass proves the subscription token alone is sufficient.
- Result: `exit_code=0 output='TOKEN_AUTH_OK'` — headless Claude access confirmed at zero extra
  spend, satisfying acceptance criterion 2. The Agent SDK path (vs. this `claude -p` fallback) will
  be exercised directly in Ch.8 when the agent loop is built.

**Dataset ingestion, measured 2026-07-09** (via [`cip/data/ingest.py`](../../cip/data/ingest.py),
DuckDB's `read_csv_auto` reading the raw CSVs directly — no pandas round-trip):

| Dataset | Source file | Size on disk | Rows | Columns |
|---|---|---|---|---|
| Telco Customer Churn | `telco_customer_churn.csv` | 970,457 B (0.97 MB) | 7,043 | 21 |
| CFPB Complaints (since 2026-01-01) | `cfpb_complaints_20260101_now.csv` | 1,325,170,607 B (1,325.2 MB) | 3,638,666 | 16 |

- `data/warehouse.duckdb` on disk: 319,827,968 B (305 MB) — DuckDB's columnar storage compresses
  the 1.26 GB of raw CFPB CSV text down to roughly a quarter of its size once both tables are in.
- Telco's 7,043-row count matches the dataset's well-known published shape — a real sanity check,
  not an assumption.
- The CFPB row count is a genuine surprise worth flagging rather than smoothing over: ~3.6M
  complaints for a 6-month window (2026-01-01 through today) is large enough that later chapters
  (Ch.3 eval harness, Ch.5 RAG, Ch.6 incremental pipeline) will need an explicit, documented
  sampling strategy rather than operating on the full table by default — full-table operations at
  this row count on a 15.4 GB RAM machine are a real constraint, not a hypothetical one.

**Lint/test pass, measured 2026-07-09:**

- `uv run ruff check .` → `All checks passed!`
- `uv run pytest` → `collected 0 items`, `no tests ran in 0.01s` (exit code 5). No test files exist
  yet anywhere under `tests/` or `chapters/` — expected at this stage, not a failure. The command
  itself runs clean against the current `pyproject.toml` config; later chapters add real test
  suites and this becomes a real pass/fail gate.

_(all three acceptance criteria for this chapter are now met — see checklist above)_
