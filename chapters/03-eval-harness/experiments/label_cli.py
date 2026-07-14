"""Human review pass over Claude's pre-labels — the step that makes the golden set golden.

Reads golden_set_prelabeled.csv (from 02_prelabel_claude.py), shows one narrative at a time with
Claude's proposed label + evidence quote, and asks the annotator to accept (Enter), override
(0/1), or mark ambiguous (a — excluded from the eval set, count reported). Every row gets human
eyes; the override rate is itself a reported measurement in this chapter's README.

Saves after every single answer (atomic write: tmp file + os.replace), so quitting anytime with
Ctrl-C or `q` never loses progress — rerunning resumes exactly where you left off.

Rubric (same as 01_sample_for_labeling.py):
    1 = narrative explicitly states the consumer closed / is closing the account, switched /
        is switching provider, or states a definite intention or threat to leave over the issue.
    0 = otherwise (complaint/dispute/anger, but no statement about ending the relationship).

Run: uv run python chapters/03-eval-harness/experiments/label_cli.py
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
PRELABELED_PATH = HERE / "golden_set_prelabeled.csv"
LABELED_PATH = HERE / "golden_set_labeled.csv"

REVIEW_COLUMNS = ["churn_signal", "ambiguous", "human_overrode"]


def load_or_init() -> pd.DataFrame:
    if LABELED_PATH.exists():
        df = pd.read_csv(LABELED_PATH, dtype={"churn_signal": "Int64"})
        df["ambiguous"] = df["ambiguous"].astype("boolean")
        df["human_overrode"] = df["human_overrode"].astype("boolean")
        return df
    if not PRELABELED_PATH.exists():
        raise SystemExit(f"{PRELABELED_PATH} missing — run 02_prelabel_claude.py first")
    df = pd.read_csv(PRELABELED_PATH)
    df["churn_signal"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df["ambiguous"] = pd.array([pd.NA] * len(df), dtype="boolean")
    df["human_overrode"] = pd.array([pd.NA] * len(df), dtype="boolean")
    return df


def save(df: pd.DataFrame) -> None:
    tmp_path = LABELED_PATH.with_suffix(".csv.tmp")
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, LABELED_PATH)


def show(position: int, total: int, row: pd.Series) -> None:
    print("\n" + "=" * 80)
    print(f"[row {position}/{total}] product: {row['product']}")
    print("=" * 80)
    print(textwrap.fill(str(row["narrative"]), width=100))
    print("-" * 80)
    verbatim = "" if row["evidence_verbatim"] else "  [!] quote NOT found verbatim in narrative"
    print(f"Claude proposes: churn_signal={row['proposed_label']} "
          f"(confidence: {row['confidence']}){verbatim}")
    evidence = str(row["evidence_quote"]) if pd.notna(row["evidence_quote"]) else ""
    if evidence:
        print(textwrap.fill(f'evidence: "{evidence}"', width=100))
    print("-" * 80)


def prompt_review(proposed: int) -> str:
    while True:
        raw = input(
            f"[Enter=accept {proposed} / 0 / 1 / a=ambiguous (excluded) / s=skip / q=quit]: "
        ).strip().lower()
        if raw in ("", "0", "1", "a", "s", "q"):
            return raw
        print("invalid input — press Enter or type 0, 1, a, s, or q")


def main() -> int:
    df = load_or_init()
    total = len(df)
    reviewed_mask = df["churn_signal"].notna() | df["ambiguous"].fillna(False)
    print(f"{int(reviewed_mask.sum())}/{total} already reviewed, "
          f"{int((~reviewed_mask).sum())} remaining\n")

    try:
        for idx in df.index[~reviewed_mask]:
            row = df.loc[idx]
            show(int(idx) + 1, total, row)
            answer = prompt_review(int(row["proposed_label"]))
            if answer == "q":
                break
            if answer == "s":
                continue
            if answer == "a":
                df.loc[idx, "ambiguous"] = True
                df.loc[idx, "human_overrode"] = False
            else:
                label = int(row["proposed_label"]) if answer == "" else int(answer)
                df.loc[idx, "churn_signal"] = label
                df.loc[idx, "ambiguous"] = False
                df.loc[idx, "human_overrode"] = label != int(row["proposed_label"])
            save(df)
    except KeyboardInterrupt:
        pass

    labeled = df["churn_signal"].notna()
    ambiguous = df["ambiguous"].fillna(False)
    n_reviewed = int((labeled | ambiguous).sum())
    print(f"\n{n_reviewed}/{total} reviewed, saved to {LABELED_PATH}")
    if n_reviewed:
        n_overrides = int(df.loc[labeled, "human_overrode"].sum())
        print(f"  labeled: {int(labeled.sum())} "
              f"(churn_signal=1: {int((df['churn_signal'] == 1).sum())}, "
              f"overrides: {n_overrides})")
        print(f"  ambiguous (excluded): {int(ambiguous.sum())}")
    if n_reviewed < total:
        print("rerun this script to resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
