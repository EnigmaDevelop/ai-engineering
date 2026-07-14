# Chapter 3 — Eval harness

CFPB complaints have no native churn label (unlike Telco in Ch.1). This chapter builds one by
hand — a real, human-verified golden set — then uses it to compare a classical baseline against a
local LLM, and to measure whether an LLM judge can be trusted to grade either one. `cip/evals/`
(golden set loader, shared metrics, judge helpers) is built here and is reused starting Ch.4.

## Goal

1. A golden set of CFPB complaint narratives labeled for an explicit **churn signal** (the
   narrative states the consumer closed/is closing the account, switched/is switching provider, or
   voices a definite intention or threat to leave) — not a proxy field, a real manual label.
2. TF-IDF + Logistic Regression vs a local zero-shot LLM (`llama3.2:3b`), scored on identical
   5-fold stratified cross-validation splits — apples-to-apples, not two different samples.
3. LLM-as-judge, with its documented biases (Zheng et al. 2023, arXiv:2306.05685) measured
   directly rather than assumed away: position bias, verbosity bias, self-preference bias,
   groundedness.
4. A real cost comparison (time, throughput, and a sampled token estimate) between the two
   classifiers.

## Method

### Golden set

`churn_signal = 1` iff the narrative **explicitly** states the consumer closed/is closing the
account, switched/is switching provider, or states a definite intention or threat to leave because
of the issue described. `churn_signal = 0` otherwise (disputing, angry, but no stated exit).

Stratified across the three CFPB products with the most usable narrative text: **Checking or
savings account**, **Credit card**, and **Debt collection** (deterministic `hash()`-ordered
sampling in [`01_sample_for_labeling.py`](experiments/01_sample_for_labeling.py) — growing a
product's sample size only extends its existing rows, never reshuffles them).

Labeling is **hybrid, not solely automated**: [`02_prelabel_claude.py`](experiments/02_prelabel_claude.py)
proposes `{label, evidence quote, confidence}` per narrative via `claude -p` (Pro subscription
OAuth, zero API spend), then every single row was reviewed and either confirmed or overridden by
the human annotator (owner) before entering the golden set — an LLM-labeled golden set would make
every later LLM evaluation circular (agreement with the labeler, not accuracy). Single annotator —
no inter-rater agreement is measurable, a real limitation, stated rather than hidden.

**Debt collection is capped at 167 rows and not grown further:** all 167 human-reviewed rows
produced zero `churn_signal=1`, for a structural reason, not an artifact of the sample — a debt
collector is an assigned relationship the consumer never chose, so "churn" from one is close to
incoherent. Further sampling there has near-zero expected return; growth was concentrated in
Checking/savings and Credit card instead (417 each).

| Product | n | positive | rate |
|---|---|---|---|
| Checking or savings account | 417 | 22 | 5.3% |
| Credit card | 417 | 26 | 6.2% |
| Debt collection | 167 | 0 | 0.0% (structural) |
| **Total** | **1001** | **48** | **4.8%** |

Human override rate over Claude's proposed label: 6/1001 (0.6%) — near-total agreement, checked
row by row, not assumed from a spot sample.

### Baseline: TF-IDF + Logistic Regression

`TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=2, stop_words="english")` +
`LogisticRegression(class_weight="balanced")` — same imbalance-handling discipline as Ch.1's
XGBoost `scale_pos_weight`, applied to LR here since positives are 4.8% of the set.

**5-fold stratified cross-validation**, not a single 70/30 split: with only 48 positives, a single
split holds out ~10 — one misclassified example swings recall by ~10 points, a false precision. CV
scores every row exactly once as a held-out case across the 5 folds, so all 48 positives
contribute, and fold-to-fold spread (mean ± std) shows how much the result actually varies instead
of hiding it behind one lucky/unlucky split. Fold assignment is saved to
[`cv_folds.json`](experiments/cv_folds.json) so the LLM classifier is scored on the **exact same**
folds. Script: [`03_baseline_tfidf.py`](experiments/03_baseline_tfidf.py).

### Local LLM: zero-shot classification

`llama3.2:3b` (project default per ADR-0001 hardware constraints), zero-shot structured output
(`format=<json schema>`, `{"churn_signal": bool, "rationale": str}`, `temperature=0`) — same
pattern as Ch.2's `03_structured_output.py`. Deliberately zero-shot only: few-shot/CoT prompting is
explicitly Ch.4's job per the roadmap.

Zero-shot has no train/test split to leak — the model classifies every row in the golden set
exactly once, and the same 5 fold assignments from the baseline are used only afterward, to
aggregate metrics the same way (mean ± std across the same 5 partitions), so the two numbers are
directly comparable. Never evaluated against Claude's own labels for classification accuracy
(Claude only proposed the golden-set labels, which a human then verified) — that would be
partially circular. Script: [`04_llm_classifier.py`](experiments/04_llm_classifier.py). Row-level
resumable (checkpoints after every call) — this environment intermittently kills long-running
background processes for reasons outside the code (confirmed via `ollama ps` that Ollama itself
stayed healthy throughout); the full 1001-row run took 50.0 minutes of actual model compute time,
spread over several hours of wall-clock time across repeated relaunches.

### LLM-as-judge bias

[`05_llm_judge_bias.py`](experiments/05_llm_judge_bias.py), four experiments using
`cip/evals/judge.py`: groundedness scoring (`qwen2.5:3b` judges `llama3.2:3b`'s rationales 1–5),
position bias (A/B order swapped for the same pair, checking whether the same candidate keeps
winning), verbosity bias (padding a rationale with content-free filler, re-scoring), self-
preference bias (both models judge both models' outputs, 2×2, checking whether a judge favors its
own model's output holding the narrative fixed).

**This bias measurement predates the golden-set growth to 1001 rows** — it ran on the earlier,
smaller labeling batch and an 8-narrative comparison sample, which is sufficient for bias
diagnostics (these measure judge behavior on paired outputs, not classifier accuracy, so they
don't need golden-set scale) but is flagged here explicitly rather than silently presented as
current-scale. Not rerun on the final 1001-row set for this chapter; also, the current
`04_llm_classifier.py` writes `results_llm_classifier.json["rows"]` as a dict keyed by
`complaint_id` (for row-level resumability) while `05_llm_judge_bias.py` still expects that field
as a list — a rerun would need that compatibility fixed first.

### Cost comparison

[`06_cost_comparison.py`](experiments/06_cost_comparison.py): TF-IDF timing is a fresh, real
fit+predict cycle on fold 0 (800 train / 201 test) of the same split used for scoring. LLM timing
reuses the actual per-row `eval_duration_ms` already recorded during the real classification run
(nothing re-run). Token counts were **not** recorded per-row during that run, so those are a
separate, explicitly-labeled **sampled estimate** (30 narratives re-sent through the identical
prompt/schema) — not a full-corpus count.

## Results

**Measured 2026-07-13/14.**

### TF-IDF vs LLM — 5-fold stratified CV (n=1001, 48 positive)

| Metric | TF-IDF + LR | LLM zero-shot (llama3.2:3b) |
|---|---|---|
| Accuracy | 0.949 ± 0.013 | 0.871 ± 0.018 |
| Precision | 0.387 ± 0.264 | 0.220 ± 0.031 |
| Recall | 0.213 ± 0.122 | **0.647 ± 0.040** |
| F1 | 0.272 ± 0.166 | **0.327 ± 0.036** |

Confusion matrix, summed across all 5 folds (every row scored exactly once):

| | TF-IDF: pred 0 | TF-IDF: pred 1 | | LLM: pred 0 | LLM: pred 1 |
|---|---|---|---|---|---|
| **actual 0** | 940 | 13 | | 841 | 112 |
| **actual 1** | 38 | 10 | | 17 | 31 |

Top TF-IDF positive-weighted terms (full-data fit, interpretability only — never used for
scoring): `close`, `close account`, `fees`, `account`, `closed`, `card`, `amex`, `account closed`,
`retirement`, `cancellation`.

### LLM-as-judge bias (earlier/smaller batch — see staleness note above)

- **Groundedness:** mean score 1.25/5 (n=61) — but this rubric turned out to be blind to
  correctly-grounded *negative* rationales (a rationale correctly saying "no exit signal present"
  scored low for not quoting an exit statement that doesn't exist), so this number understates
  actual groundedness rather than measuring judge quality. Documented as a rubric design flaw, not
  a finding about the model.
- **Position bias:** 87.5% consistency (7/8) across A/B swap — i.e., in 1/8 cases the "winner"
  flipped purely because of which position it was shown in.
- **Self-preference:** 11/16 judgments (forward) favored the judge's own model; almost identical
  to the swap-adjusted rate, meaning the effect measured is overwhelmingly **position bias wearing
  a self-preference costume**, not genuine self-preference.
- **Verbosity bias:** 0/6 padded rationales scored higher after content-free filler was added —
  no verbosity effect detected in this sample.

### Cost comparison

| | TF-IDF + LR | LLM zero-shot (llama3.2:3b) |
|---|---|---|
| Fit + predict (fold 0: 800 train / 201 test) | 0.203 s total | — |
| Throughput | 4,943 rows/s (full pipeline) | 0.333 rows/s |
| Per-row cost | ~0.2 ms | ~3.0 s (± 1.4 s) |
| Tokens per call | n/a | ~393 prompt + ~33 response (sampled, n=30) |

**TF-IDF is ~14,844× faster per row than the LLM**, on this CPU-bound hardware (15.4 GB RAM, Intel
Arc iGPU 2 GB VRAM — ADR-0001). TF-IDF+LR has no model weights to load and runs entirely on CPU;
the LLM pays a real per-call cost that TF-IDF structurally cannot incur.

## Decision

- **No universal winner — the two models trade precision for recall.** TF-IDF is precise but
  misses most positives (recall 0.213); the LLM catches 3× more true positives (recall 0.647) at
  the cost of far more false positives (112 vs 13). Which matters more depends on what the
  classification feeds into: a system that routes flagged complaints to a human for review can
  tolerate the LLM's false-positive rate to avoid missing real churn signals; a system that acts
  automatically on the label cannot.
- **The LLM is also the more stable estimator across folds**, not just the higher-recall one — its
  metric std is 0.02–0.04 across the board, while TF-IDF's precision std is 0.264, meaning TF-IDF's
  headline precision number is highly sensitive to which 10 positives happen to land in the test
  fold. On a rare positive class, that stability is worth reporting alongside the point estimate,
  not just the point estimate itself.
- **Accuracy is not the metric to lead with here** — TF-IDF's 0.949 accuracy on a 95.2%-negative
  set is the same trap Ch.1 flagged for the majority-class baseline; it moves in the opposite
  direction of the metric (recall) that actually reflects the task.
- **Cost is not a close call**: ~14,844× per-row throughput difference means TF-IDF is the correct
  default at any scale where the LLM's recall advantage doesn't clearly justify it, and the LLM is
  only worth its cost where the review pipeline can absorb its false-positive rate.
- **The judge-bias measurement needs a rerun disclaimer, not a silent reuse** — it predates the
  golden set's growth to 1001 rows and was run on a smaller batch; the finding that stands
  (position bias, not genuine self-preference, drove the apparent self-preference signal) is a
  property of the judging mechanism, not of scale, so it's reported as-is rather than re-measured
  for this chapter — but a future rerun requires first fixing `05_llm_judge_bias.py`'s assumption
  that `results_llm_classifier.json["rows"]` is a list (it's now a dict, for resumability).
  Groundedness's low mean score is flagged as a rubric flaw, not restated as a real finding about
  rationale quality.
- **Schema field order (rationale-first vs decision-first) was scoped but not run.** Estimated
  cost: ~50 minutes of model compute for a full second pass over all 1001 rows (matching the
  measured 3.0 s/row rate), with realistic wall-clock risk of several hours given this
  environment's background-process interruptions — or ~5–8 minutes on a targeted ~100–150-row
  subsample if only testing the post-hoc-rationalization hypothesis. Deferred; not part of this
  chapter's committed results.

## Checklist

- [x] Golden set: 1001 CFPB narratives, hybrid Claude-propose + 100% human-reviewed, rubric
      documented, single-annotator limitation stated
- [x] Debt collection's zero-positive result treated as a structural finding (documented), not
      grown further past 167
- [x] TF-IDF + LR and LLM zero-shot scored on **identical** 5-fold stratified CV splits
      (`cv_folds.json` shared between both scripts)
- [x] Full metric set (accuracy/precision/recall/F1/confusion matrix) via shared
      `cip/evals/metrics.py`, mean ± std across folds, not a single point estimate
- [x] LLM-as-judge biases measured directly (position, verbosity, self-preference, groundedness),
      including a documented rubric flaw and a swap-control that separates genuine self-preference
      from pure position bias
- [x] Real, measured cost comparison (timing + throughput); token counts explicitly labeled as a
      sampled estimate, not a full-corpus count
- [x] Results committed as JSON under `experiments/`
- [x] `uv run pytest` / `uv run ruff check .` clean (verified 2026-07-14: ruff `All checks
      passed!`; pytest 9 passed)
- [ ] Schema field-order experiment (rationale-first vs decision-first) — scoped, cost estimated,
      deferred by owner decision
- [ ] LLM-as-judge bias rerun on the final 1001-row set — deferred; needs `05_llm_judge_bias.py`
      updated for `04`'s dict-shaped `rows` field first
