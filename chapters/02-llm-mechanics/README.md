# Chapter 2 — LLM mechanics, locally

Still no prompt engineering in the "ask an LLM to do my job" sense — this chapter measures the
mechanics underneath one: tokenization, sampling parameters, structured output, and quantization, all
run on local Ollama models on this project's actual hardware (15.4 GB RAM, Intel Arc iGPU with 2 GB
VRAM — CPU-bound inference, see [ADR-0001](../../docs/adr/0001-tech-stack.md)).

## Goal

Four mechanisms, each measured rather than assumed: tokenizer behavior on real CFPB complaint text,
temperature/top-p's effect on output determinism and diversity, JSON-schema-constrained structured
output (compliance rate and, critically, its limits), and the size/latency/quality tradeoff across
three quantization levels of the same model.

## Method

- **Shared client:** [`cip/llm/ollama_client.py`](../../cip/llm/ollama_client.py) — a thin `requests`
  wrapper around Ollama's REST API (`/api/generate`, `/api/tags`, `/api/show`), returning parsed
  timing fields (`load_duration`, `prompt_eval_count`, `eval_count`, `eval_duration`) alongside the
  response text. No SDK dependency; Ollama's HTTP surface is small enough not to need one.
- **Data:** real CFPB complaint narratives from `data/warehouse.duckdb` (Ch.0) — never synthetic
  text — for every experiment that needs prompt content.
- **01 — Tokenization:** `prompt_eval_count` from a real (`raw=True`, no chat template) and templated
  (`raw=False`) call, across `llama3.2:3b` and `qwen2.5:3b`, on 5 real complaint narratives plus one
  short probe string.
- **02 — Sampling:** one real narrative, summarized 8 times per setting across
  `temperature ∈ {0, 0.3, 0.7, 1.0, 1.5}` and `top_p ∈ {0.1, 0.5, 0.9, 1.0}` (temperature fixed at 0.8
  for the top_p sweep). Diversity measured two ways: exact-match `distinct_outputs` and
  `avg_pairwise_jaccard` (word-set overlap across all pairs).
- **03 — Structured output:** `format=<json schema>` requiring `{product, issue_summary, urgency,
  mentions_legal_action}`, against 15 real narratives with known CFPB `Product` labels. Measures
  `json_parse_rate`, `schema_valid_rate`, and an explicitly-scoped-as-informal
  `loose_product_match_rate` (substring check against the real CFPB label — not a rigorous eval;
  Ch.3 builds the real golden-set/LLM-as-judge harness).
- **04 — Quantization:** `llama3.2:3b` pulled at `Q4_K_M` (default), `Q8_0`, and `fp16`. Cold-load
  latency measured by unloading (`keep_alive=0`) before each timing so the load is genuinely cold;
  quality measured by reusing 03's structured-extraction task and its two proxy rates, across the
  same 8 narratives, per quant level.

Scripts: [`experiments/01_tokenization.py`](experiments/01_tokenization.py),
[`02_sampling.py`](experiments/02_sampling.py),
[`03_structured_output.py`](experiments/03_structured_output.py),
[`04_quantization.py`](experiments/04_quantization.py). Raw results: `experiments/results_*.json`.

## Results

**Measured 2026-07-09.**

**Tokenization** — same text, different tokenizer, different count (`raw=True`, no template):

| Complaint | Chars | llama3.2:3b tokens | qwen2.5:3b tokens |
|---|---|---|---|
| 18398333 | 527 | 103 | 108 |
| 18398392 | 430 | 96 | 102 |
| 18398383 | 264 | 57 | 57 |

Chat-template tax on a 6-word probe (tokens spent before any real content): llama3.2:3b **24**,
qwen2.5:3b **29**.

**Sampling** — `llama3.2:3b`, 8 repeats/setting:

| temperature | distinct/8 | avg pairwise Jaccard | top_p (temp=0.8) | distinct/8 | avg pairwise Jaccard |
|---|---|---|---|---|---|
| 0.0 | 1 | 1.000 | 0.1 | 1 | 1.000 |
| 0.3 | 8 | 0.516 | 0.5 | 8 | 0.595 |
| 0.7 | 8 | 0.386 | 0.9 | 8 | 0.366 |
| 1.0 | 8 | 0.342 | 1.0 | 8 | 0.374 |
| 1.5 | 8 | 0.315 | | | |

`temperature=0` produced the byte-identical output all 8 times. Jaccard falls monotonically from
0.3 → 1.5. `top_p=0.1` shows the same collapse-to-deterministic effect. `top_p` 0.9 → 1.0 does not
continue the downward trend (0.366 → 0.374) — a small, real, non-monotonic step at the high end.

**Structured output** — `llama3.2:3b`, 15 real narratives:

| Metric | Value |
|---|---|
| JSON parse rate | 1.000 |
| Schema-valid rate | 1.000 |
| Loose product-match rate (informal) | 0.067 (1/15) |
| Mean eval duration | 4,289 ms |

**Quantization** — `llama3.2:3b` at 3 quant levels, 8 real narratives for the quality proxy:

| Quant | Disk | Cold load | Warm load | Tok/s | Schema-valid | Loose match (n=8) |
|---|---|---|---|---|---|---|
| Q4_K_M | 2.02 GB | 4.07 s | 0.50 s | 16.2 | 1.0 | 0.125 |
| Q8_0 | 3.42 GB | 6.69 s | 0.51 s | 12.7 | 1.0 | 0.000 |
| fp16 | 6.43 GB | 22.84 s | 0.61 s | 7.1 | 1.0 | 0.000 |

## Decision

- **Tokenizer vocabularies are not portable across model families** — the same real complaint text
  yields different token counts on Llama vs. Qwen tokenizers, and chat templates add a fixed
  24–29-token cost on every call regardless of content. Any later chapter reasoning about context
  budgets or per-call cost needs to measure this per model, not assume a universal
  characters-per-token ratio.
- **Sampling parameters behave exactly as documented, with one honest non-monotonic wrinkle**:
  `temperature=0` and `top_p=0.1` both collapse to deterministic output; diversity increases with
  temperature up to 1.5. `top_p` 0.9→1.0 does not continue the downward diversity trend — reported as
  a real measurement, not smoothed into a clean story it didn't produce.
- **Grammar-constrained structured output is a shape guarantee, not a correctness guarantee** — this
  chapter's single most important finding. 100% schema-valid JSON, but only 6.7% of extractions used
  CFPB's actual product-category vocabulary, because the schema never communicated that a controlled
  vocabulary existed. Any future chapter using `format=<schema>` must still validate semantic content
  against ground truth (Ch.3's job), not treat schema-validity as a proxy for accuracy.
- **Quantization's size/latency tradeoff is real, large, and monotonic on this hardware**: fp16 is
  3.2× the disk size of Q4_K_M and costs 5.6× the cold-load time, at less than half the tokens/sec.
  The quality proxy showed no measurable advantage for higher precision at `n=8` — reported as
  underpowered-to-detect, not as evidence that quantization is quality-free; a real quality
  comparison needs Ch.3's eval harness on a larger set.
- **Every measured number here needs Ch.3 to become a real accuracy claim.** This chapter deliberately
  stops at "does the shape hold, how fast is it, how random is it" — none of the quality proxies used
  here (`loose_product_match_rate`) are meant to be quoted as accuracy; they're sanity signals that
  motivate why Ch.3's golden-set/LLM-as-judge harness is the next necessary piece of infrastructure.

## Checklist

- [x] Shared Ollama REST client in `cip/llm/ollama_client.py`
- [x] Tokenization measured across 2 model families on real CFPB text + chat-template tax quantified
- [x] Temperature and top_p sweeps measured with two independent diversity signals
- [x] Structured output measured for schema compliance AND its semantic limits, on real data with
      real ground-truth labels
- [x] Quantization measured across 3 real quant levels (Q4_K_M/Q8_0/fp16) on disk size, cold/warm
      load latency, tokens/sec, and a quality proxy
- [x] All raw results committed as JSON under `experiments/`
- [x] `uv run pytest` / `uv run ruff check .` clean (verified 2026-07-09: ruff `All checks passed!`;
      pytest collects 0 items — no test suites exist yet, same expected state as Ch.0/Ch.1)
