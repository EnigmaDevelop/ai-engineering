# ADR-0001: Tech stack and repository architecture

Date: 2026-07-09 · Status: accepted

## Context

Portfolio repo built by one person, ~30 h/week, with three hard constraints:

1. **Zero extra spend.** Claude Pro subscription exists; the pay-per-token Claude API does not.
2. **Privacy as a theme.** Local inference ("data never leaves the machine") is a deliberate
   portfolio narrative, not just a cost measure.
3. **Measured hardware** (checked 2026-07-09 on the dev machine): 15.4 GB RAM, Intel Arc iGPU
   with 2 GB VRAM, Python 3.13.0, git 2.47, Ollama 0.31.1 (outdated — structured outputs require
   ≥0.5.x), Claude Code 2.1.205.

## Decisions

| Area | Decision | Rationale |
|------|----------|-----------|
| Package/env manager | **uv** | Single tool for venv, lockfile, Python versions; fast; current community default. |
| Analytics storage | **DuckDB + Parquet** | Zero-ops lakehouse on a laptop; SQL surface matches the owner's data-platform background; scales to the CFPB dataset (millions of rows). |
| Local LLM runtime | **Ollama** | Free, private, OpenAI-compatible API for raw LLM mechanics. Default models sized to hardware: **3–4B** (llama3.2:3b, qwen2.5:3b); 7–8B Q4 allowed but expected CPU-slow — to be measured in Ch.2, not assumed. |
| Frontier model access | **Claude Agent SDK / `claude -p` via Pro subscription OAuth** (`claude setup-token`) | Only no-extra-cost path to Claude. To be verified in Ch.0; fallback is headless `claude -p`. API-only features (batch API, caching billing) are covered conceptually with official docs as evidence. |
| Shared code package | **`cip/`** (Customer Intelligence Platform) | The plan's original name `platform/` shadows Python's stdlib `platform` module — scripts run from repo root would import the wrong module. Renamed to avoid the clash. |
| Repo shape | **Monorepo: shared `cip/` core + `chapters/NN-*/`** | Chapters must read as one platform evolving, not 13 demos; shared eval harness and data layer are reused across chapters. |
| Data | **Real public datasets only** (IBM Telco Churn, CFPB Consumer Complaints) | Owner's evidence principle: no synthetic anecdotes. CFPB updates monthly → genuine incremental-load problem for Ch.6. |

## Consequences

- Anything requiring >8B local models or paid API experiments is out of scope; articles must say
  so explicitly rather than extrapolate.
- Ollama must be upgraded before Ch.2 (structured outputs).
- `uv sync` installs `cip` editable, so `from cip.data import ...` works everywhere including
  pytest and chapter scripts.
- Claude Pro rate limits (5-hour and weekly windows) are a real scheduling constraint at
  30 h/week; implementation runs on Sonnet, heavy experiments run on Ollama.
