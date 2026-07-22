"""Document assembly for the Lost-in-the-Middle experiment — born in Chapter 4.

Filler is always real CFPB narratives (never invented text), drawn from confirmed-negative rows
(churn_signal=0) so there is no risk of the filler itself containing the signal being tested for.
For a target row and a desired filler word budget, this samples real filler docs up to that budget
with a FIXED seed per (target, budget) pair, then places the target document at the requested
position — the same filler set and order is reused across all three positions for a given target,
so moving only the target's position is a real paired comparison, not three independent draws.

Documents are never labeled "distractor" in the prompt (see prompts.py's build_prompt_litm) —
only real complaint_id-derived ids are used.
"""

from __future__ import annotations

import random

import pandas as pd

# ~1.3 tokens/word observed in Ch.3/Ch.4 calibration (394 prompt tokens over ~309 rubric+narrative
# +instruction words) — used only to size the filler draw approximately; the real token count is
# always read back from Ollama's prompt_eval_count after the call, never trusted as exact upfront.
TOKENS_PER_WORD_APPROX = 1.3


def build_filler_pool(df: pd.DataFrame, product: str | None = None) -> list[dict]:
    """Real negative-labeled narratives with real complaint-id-derived doc ids."""
    neg = df[df["churn_signal"] == 0]
    if product is not None:
        neg = neg[neg["product"] == product]
    return [
        {"id": f"filler_{row['complaint_id']}", "text": row["narrative"]}
        for _, row in neg.iterrows()
    ]


def assemble_context(
    target_id: str,
    target_text: str,
    filler_pool: list[dict],
    position: str,
    context_token_budget: int,
    seed: int,
) -> list[dict]:
    """position: "start" | "middle" | "end"."""
    target_words = len(target_text.split())
    filler_word_budget = max(
        0, int(context_token_budget / TOKENS_PER_WORD_APPROX) - target_words
    )

    rng = random.Random(seed)
    pool_copy = filler_pool.copy()
    rng.shuffle(pool_copy)

    filler_docs = []
    word_count = 0
    for doc in pool_copy:
        if word_count >= filler_word_budget:
            break
        filler_docs.append(doc)
        word_count += len(doc["text"].split())

    target_doc = {"id": target_id, "text": target_text}
    if position == "start":
        return [target_doc, *filler_docs]
    if position == "end":
        return [*filler_docs, target_doc]
    mid = len(filler_docs) // 2
    return [*filler_docs[:mid], target_doc, *filler_docs[mid:]]
