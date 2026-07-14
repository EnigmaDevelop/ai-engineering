# ADR-0002: Publish the Ch.3 golden set on Kaggle

Date: 2026-07-14 · Status: accepted

## Context

Chapter 3 produced a 1,001-row, human-verified churn-signal golden set on real CFPB Consumer
Complaint Database narratives — every label proposed by Claude and then confirmed or overridden
by a human annotator (6/1001 overrides), plus committed model predictions (TF-IDF+LR, local
zero-shot LLM) and the canonical 5-fold CV split used to score them. A hand-verified LLM-eval
golden set with paired model predictions is uncommon in public data catalogs (most published NLP
sets are scraped or synthetically labeled), and this repo's portfolio goal explicitly includes
public-facing evidence of the work, not just private artifacts.

**Source-data license basis** (verified 2026-07-14 against CFPB's own site):
- consumerfinance.gov's Consumer Complaint Database page states: *"All complaint data we publish
  is freely available for anyone to use, analyze, and build on."*
- The same page confirms narratives are published *"if the consumer opts to share it publicly
  and after the Bureau takes steps to remove personal information"* — consent and scrubbing are
  CFPB's own process, not something this project performs.
- CFPB is a U.S. federal agency; under 17 U.S.C. § 105, works of the U.S. government are not
  subject to domestic copyright protection. Combined with the page's explicit "freely available
  ... use, analyze, and build on" statement, republishing the narratives (as already done,
  scrubbed, by CFPB) is on solid ground.

## Decision

Publish the golden set on Kaggle as `cfpb-churn-signal-golden-set` under **CC BY 4.0**: the
underlying CFPB narrative text is public domain per the above, but the annotation layer this
project adds (`churn_signal` labels, AI-proposal/human-override provenance, model predictions,
rationales, fold assignments) is original work this project licenses under attribution terms —
consistent with the portfolio goal of visible, attributable work.

Alongside the dataset, publish one tutorial-style starter notebook
(`churn-signal-tfidf-vs-llm.ipynb`) that reproduces this chapter's headline comparison using only
the shipped CSVs — training TF-IDF live (sub-second) and reading the LLM's already-generated
predictions — so nobody needs a GPU, Ollama, or an API key to explore the result.

Both artifacts are generated, not hand-edited: `kaggle/churn-signal-golden-set/build_dataset.py`
regenerates all four dataset CSVs deterministically from the chapter's already-committed
artifacts (`golden_set_labeled.csv`, `results_llm_classifier.json`, `results_baseline_tfidf.json`,
`cv_folds.json`). No LLM or API call is made to build the Kaggle release — every number it ships
was already measured and committed in Chapter 3.

## Consequences

- **Contamination risk, accepted deliberately.** Publishing an eval set makes it a plausible
  future training-data ingredient; over time this could erode its value as an uncontaminated
  benchmark for churn-signal detection. Mitigation: the dataset card states the intended use is
  evaluation, not training, and asks readers not to train on it — a request, not a technical
  guarantee. This trade-off is accepted in exchange for portfolio visibility; it would not be
  accepted for a benchmark this project depended on staying uncontaminated long-term.
- **Maintenance duty.** When the companion Medium article (Ch.3's `ARTICLE.md`, drafted privately
  per `CLAUDE.local.md`) is published, the dataset description and notebook intro get a
  `kaggle datasets version` / kernel re-push adding the article link — tracked as a follow-up,
  not done at initial publish time (the article isn't live yet).
- **Single point of truth stays the GitHub repo.** The Kaggle card links back to
  `chapters/03-eval-harness/README.md` for full methodology rather than duplicating it, so the
  repo (not the Kaggle listing) remains the canonical source if the two ever diverge.
- **License scope is explicit, not assumed.** CC BY 4.0 applies to this project's added
  annotations and derived artifacts; it does not purport to re-license CFPB's own narrative text,
  which is already public domain on its own terms.
