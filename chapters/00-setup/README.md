# Chapter 0 — Setup

No article for this chapter; it exists so every later chapter starts from a verified foundation.

## Goal

Working environment + both datasets ingested into DuckDB + **proof** that Claude is reachable at
zero extra cost through the Pro subscription.

## Checklist

- [x] git repo, uv project, `cip/` package skeleton (installed editable via `uv sync`)
- [x] Hardware measured and recorded in [ADR-0001](../../docs/adr/0001-tech-stack.md):
      15.4 GB RAM, Intel Arc iGPU 2 GB VRAM → default local models 3–4B
- [ ] Upgrade Ollama (installed: 0.31.1; structured outputs need ≥0.5.x) and pull
      `llama3.2:3b`, `qwen2.5:3b`, `nomic-embed-text`
- [ ] `claude setup-token` → verify the Agent SDK (or `claude -p` headless) works with the Pro
      subscription OAuth token; record the working example under `experiments/`
- [ ] `uv run python -m cip.data.download --dataset telco`
- [ ] `uv run python -m cip.data.download --dataset cfpb --since 2026-01-01`
- [ ] Ingest both into DuckDB (`data/warehouse.duckdb`) with row counts recorded below

## Acceptance criteria

1. A committed note in this README with the **measured row counts** of both datasets.
2. A runnable script under `experiments/` proving Claude access with subscription auth. If the
   Agent SDK refuses subscription auth, the fallback (`claude -p`) is documented here and the
   plan's Claude chapters are revised accordingly.
3. `uv run pytest` and `uv run ruff check .` pass clean.

## Results

_(to be filled with measurements — no placeholders, no estimates)_
