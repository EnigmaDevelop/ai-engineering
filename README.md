# AI Engineering — from Data Platforms to Production LLM Systems

A chapter-by-chapter portfolio documenting a deliberate transition: 9 years of data platform work
(churn analytics, data science, BI, data engineering) applied to AI engineering. Every chapter
ships working code, **reproducible measurements**, an architecture decision record, and a
published article on [Level Up Coding](https://levelup.gitconnected.com/).

**One domain, one evolving platform.** Instead of 13 disconnected demos, every chapter builds on
the same customer-analytics foundation — real public data, a real data platform underneath — and
ends in a capstone: a **Customer Intelligence Platform**.

## Principles

1. **Evidence over claims.** Every number in every article is reproducible from `experiments/` in
   the corresponding chapter. No hypothetical scenarios, no synthetic anecdotes.
2. **Baselines first.** LLM approaches are always measured against the boring classical baseline.
3. **Local-first, privacy-aware.** Raw LLM experiments run on local models (Ollama) — the data
   never leaves the machine. Frontier-model work uses the Claude Agent SDK.
4. **The pipeline is the product.** AI features sit on top of data engineering that is incremental,
   idempotent, and observable.

## Data

Real, public datasets only:

- [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) — tabular churn baseline.
- [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) —
  millions of real customer complaint narratives, updated monthly (a genuine incremental-load problem).

## Roadmap

| # | Chapter | Focus | Status |
|---|---------|-------|--------|
| 0 | [Setup](chapters/00-setup/) | Environment, data ingestion, access verification | ✅ done |
| 1 | [Baseline first](chapters/01-baseline/) | XGBoost churn model — the anchor every LLM result is measured against | ✅ done |
| 2 | [LLM mechanics, locally](chapters/02-llm-mechanics/) | Tokenization, sampling, structured output, quantization — measured on consumer hardware | ✅ done |
| 3 | Eval harness | Golden sets, LLM-as-judge, LLM vs TF-IDF on real complaints | ⏳ |
| 4 | Prompt & context engineering | Few-shot / CoT / context-position effects, all measured | ⏳ |
| 5 | RAG through a data engineer's lens | Chunking A/B, retrieval metrics, vector store comparison | ⏳ |
| 6 | Production data engineering for AI | Incremental, idempotent embedding pipelines; contracts | ⏳ |
| 7 | Semantic layer + text-to-SQL | LLM-generated SQL scored against a golden query set | ⏳ |
| 8 | Tool use & the agent loop | ReAct from scratch, then the Claude Agent SDK | ⏳ |
| 9 | Subagents & orchestration | Orchestrator–worker, context isolation — measured | ⏳ |
| 10 | Automation & LLMOps | Eval gates in CI, scheduled agents, prompt versioning | ⏳ |
| 11 | Performance optimization | Semantic caching, model routing, quantization deep-dive | ⏳ |
| 12 | Observability & guardrails | Tracing, token/latency dashboards, injection defenses | ⏳ |
| 13 | Capstone: Customer Intelligence Platform | Everything integrated, reference architecture | ⏳ |

## Repository layout

```
cip/            shared platform core (data, llm, evals, observability) — grows chapter by chapter
chapters/       one directory per chapter: README (results), src, experiments
docs/adr/       architecture decision records
data/           raw data (gitignored) + small committed samples
```

> Why `cip/` and not `platform/`? `platform` is a Python standard-library module;
> shadowing it causes subtle import bugs. See [ADR-0001](docs/adr/0001-tech-stack.md).

## Running anything

```
uv sync                          # create the environment
uv run python -m cip.data.download --dataset telco
uv run pytest                    # tests
```

Each chapter's README documents its own `uv run` experiment commands.
