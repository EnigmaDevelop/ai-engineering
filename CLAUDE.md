# CLAUDE.md

Portfolio repo: AI engineering learning-in-public, chapter by chapter. Owner has 9 years of data
platform experience. Working language with the owner is Turkish; all repo content and code are
in English.

## Non-negotiable rules

1. **Evidence standard.** Every claim needs a measurement, an official doc reference, or a code
   reference. Never invent hypothetical scenarios or synthetic anecdotes. If something cannot be
   verified here, mark it explicitly as "needs verification".
2. **Real data only.** Datasets: IBM Telco Churn, CFPB Consumer Complaints. Do not generate
   synthetic data to fill gaps.
3. **Zero extra spend.** No pay-per-token Claude API calls. Claude access goes through the Pro
   subscription (Claude Agent SDK / `claude -p` with `claude setup-token` OAuth). Raw LLM
   experiments run on Ollama locally.
4. **Educational answers.** The owner asks many questions while working — answer them as a
   teacher: explain the mechanism, cite the source (doc/paper/measurement), connect it to their
   data-platform background.
5. **Architecture decisions go to `docs/adr/`** (sequential numbering, context → decision →
   consequences).

## Engineering conventions

- Python ≥3.12 managed by uv. Run everything via `uv run ...`. Lint: `uv run ruff check .`
  Tests: `uv run pytest`.
- Shared code lives in `cip/` (never name a package `platform` — stdlib clash, see ADR-0001).
- Experiments must be reproducible: a script under `chapters/NN-*/experiments/` + raw results
  committed as CSV/JSON. Every published number must be regenerable from the repo.
- Chapter READMEs follow: goal → method → results table → decision.
- Hardware context for local models: 15.4 GB RAM, Intel Arc iGPU (2 GB VRAM) → default to 3–4B
  models (e.g. llama3.2:3b, qwen2.5:3b); 7–8B Q4 is possible but slow (CPU-bound). Measure, don't
  assume.
