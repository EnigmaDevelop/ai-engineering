"""Unit tests for Lost-in-the-Middle document assembly (litm_documents.py)."""

import pandas as pd

from litm_documents import assemble_context, build_filler_pool


def make_df() -> pd.DataFrame:
    rows = [
        {"complaint_id": i, "product": "Debt collection", "narrative": f"filler text {i} " * 10,
         "churn_signal": 0}
        for i in range(20)
    ]
    rows.append({"complaint_id": 999, "product": "Credit card",
                 "narrative": "I closed my account.", "churn_signal": 1})
    return pd.DataFrame(rows)


def test_build_filler_pool_excludes_positives_and_filters_product():
    df = make_df()
    pool = build_filler_pool(df, product="Debt collection")
    assert len(pool) == 20
    assert all(doc["id"].startswith("filler_") for doc in pool)


def test_assemble_context_places_target_at_requested_position():
    df = make_df()
    pool = build_filler_pool(df, product="Debt collection")

    start_docs = assemble_context("target", "the target text", pool, "start", 500, seed=1)
    end_docs = assemble_context("target", "the target text", pool, "end", 500, seed=1)
    middle_docs = assemble_context("target", "the target text", pool, "middle", 500, seed=1)

    assert start_docs[0]["id"] == "target"
    assert end_docs[-1]["id"] == "target"
    middle_index = [d["id"] for d in middle_docs].index("target")
    assert 0 < middle_index < len(middle_docs) - 1


def test_assemble_context_reuses_identical_filler_across_positions():
    df = make_df()
    pool = build_filler_pool(df, product="Debt collection")

    start_docs = assemble_context("target", "the target text", pool, "start", 500, seed=7)
    end_docs = assemble_context("target", "the target text", pool, "end", 500, seed=7)

    start_filler_ids = {d["id"] for d in start_docs if d["id"] != "target"}
    end_filler_ids = {d["id"] for d in end_docs if d["id"] != "target"}
    assert start_filler_ids == end_filler_ids


def test_assemble_context_respects_word_budget_roughly():
    df = make_df()
    pool = build_filler_pool(df, product="Debt collection")
    docs = assemble_context("target", "short", pool, "start", context_token_budget=130, seed=1)
    # ~130 tokens / 1.3 tokens-per-word ~= 100 words budget -> a handful of filler docs, not all 20
    assert 0 < len(docs) < 21
