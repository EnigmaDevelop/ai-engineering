"""LLM-as-judge, with its own documented biases measured rather than assumed away.

Zheng et al. 2023 ("Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", arXiv:2306.05685)
legitimizes LLM judging by validating against human labels, but also documents systematic judge
biases: position bias (preferring whichever candidate is shown first), verbosity bias (preferring
longer answers regardless of content), and self-enhancement bias (a model favoring its own
output). This script replicates all three, locally, using the two models this project already
runs (llama3.2:3b, qwen2.5:3b), plus a groundedness-scoring pass over 04's actual classifier
rationales.

Four experiments, using cip.evals.judge (built on cip.llm.ollama_client, same structured-output
pattern as Ch.2):

  1. Groundedness  — qwen2.5:3b scores every llama3.2:3b rationale from 04_llm_classifier.py's
     test-set predictions, 1-5, for whether it's actually grounded in the narrative.
  2. Position bias — for a sample of narratives, both models generate a rationale; a judge
     compares the pair in both orders (A/B swapped) and we measure how often the SAME candidate
     wins regardless of which position it was shown in.
  3. Verbosity bias — a subset of rationales get padded with content-free filler (same facts,
     more words); re-scored for groundedness to see if padding alone raises the score.
  4. Self-preference — both models judge both models' rationales (2x2), measuring whether a
     judge scores its own model's output higher than the other model's, holding narratives fixed.

Calls are batched BY MODEL (all llama3.2:3b calls, then all qwen2.5:3b calls, per phase) to
avoid the reload contention this hardware hits when swapping loaded models on every call
(CPU-bound, 2 GB iGPU VRAM — ADR-0001).

Fully row-level resumable: every single judge/generation call is checkpointed to
results_llm_judge_bias_partial.json immediately after it completes. This environment has been
killing long-running background processes after well under a minute for reasons outside this
script (server-side session/session-sandbox lifetime, not this code or the Ollama server, which
responds fine) — rerunning this script picks up from the last completed call instead of losing
all progress and starting over.

Run: uv run python chapters/03-eval-harness/experiments/05_llm_judge_bias.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

import requests

from cip.evals.golden_set import load_golden_set
from cip.evals.judge import compare_candidates, score_groundedness
from cip.llm.ollama_client import generate

HERE = Path(__file__).parent
GOLDEN_SET_PATH = HERE / "golden_set_labeled.csv"
CLASSIFIER_RESULTS_PATH = HERE / "results_llm_classifier.json"
RESULTS_PATH = HERE / "results_llm_judge_bias.json"
PARTIAL_PATH = HERE / "results_llm_judge_bias_partial.json"

MODEL_A = "llama3.2:3b"
MODEL_B = "qwen2.5:3b"
JUDGE_FOR_GROUNDEDNESS = "qwen2.5:3b"  # different family from the classifier under scrutiny

N_COMPARISON_SAMPLE = 8  # narratives used for position-bias / self-preference
N_VERBOSITY_SAMPLE = 6   # subset of 04's rationales padded and re-scored
RETRY_ATTEMPTS = 3
RETRY_DELAY_S = 8

FILLER = (
    " This is a significant matter that deserves careful attention and should be reviewed "
    "thoroughly given its potential impact on the consumer's overall financial situation."
)

RATIONALE_PROMPT = (
    "You classify US consumer complaint narratives for an EXPLICIT signal that the consumer is "
    "ending their relationship with the company (closed/closing the account, switched/switching "
    "provider, or a definite intention or threat to leave). Whether or not that signal is "
    "present, write ONE sentence explaining your reasoning, grounded in the narrative text.\n\n"
    "NARRATIVE:\n{narrative}\n\nReturn only the one-sentence rationale, no other text."
)

T = TypeVar("T")


def with_retry(fn: Callable[[], T], label: str) -> T:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except requests.exceptions.RequestException as exc:
            if attempt == RETRY_ATTEMPTS:
                raise
            print(f"    [retry {attempt}/{RETRY_ATTEMPTS - 1}] {label}: {exc!r}", flush=True)
            time.sleep(RETRY_DELAY_S)
    raise RuntimeError("unreachable")  # pragma: no cover


class Checkpoint:
    """Row-level resumable state, flushed to disk after every single call."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.exists() else {
            "groundedness": {},       # complaint_id(str) -> {score, reasoning}
            "rationale_a": {},        # complaint_id(str) -> rationale text (llama)
            "rationale_b": {},        # complaint_id(str) -> rationale text (qwen)
            "comparisons": {MODEL_A: {}, MODEL_B: {}},  # judge -> complaint_id -> {forward, swapped}
            "verbosity": {},          # complaint_id(str) -> {padded_score}
        }

    def save(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2))


def run_groundedness(ck: Checkpoint, rows: list[dict]) -> None:
    todo = [r for r in rows if not r["parse_failed"] and r["rationale"]
            and str(r["complaint_id"]) not in ck.data["groundedness"]]
    print(f"  {len(todo)} remaining (of {len(rows)})", flush=True)
    for row in todo:
        judged = with_retry(
            lambda r=row: score_groundedness(JUDGE_FOR_GROUNDEDNESS, r["narrative"], r["rationale"]),
            f"groundedness({row['complaint_id']})",
        )
        ck.data["groundedness"][str(row["complaint_id"])] = judged
        ck.save()
        print(f"  groundedness {row['complaint_id']}: {judged.get('score')}", flush=True)


def run_rationales(ck: Checkpoint, model: str, key: str, narratives: list[dict]) -> None:
    todo = [n for n in narratives if str(n["complaint_id"]) not in ck.data[key]]
    print(f"  {model}: {len(todo)} remaining (of {len(narratives)})", flush=True)
    for item in todo:
        rationale = with_retry(
            lambda i=item: generate(
                model, RATIONALE_PROMPT.format(narrative=i["narrative"]), options={"temperature": 0}
            ).response.strip(),
            f"rationale({model},{item['complaint_id']})",
        )
        ck.data[key][str(item["complaint_id"])] = rationale
        ck.save()
        print(f"    {item['complaint_id']}: {rationale[:80]!r}", flush=True)


def run_comparisons(ck: Checkpoint, judge: str, narratives: list[dict]) -> None:
    done = ck.data["comparisons"][judge]
    todo = [n for n in narratives if str(n["complaint_id"]) not in done]
    print(f"  judge={judge}: {len(todo)} remaining (of {len(narratives)})", flush=True)
    for item in todo:
        cid = str(item["complaint_id"])
        a = ck.data["rationale_a"][cid]
        b = ck.data["rationale_b"][cid]
        forward = with_retry(
            lambda: compare_candidates(judge, item["narrative"], a, b),
            f"compare_forward(judge={judge},{cid})",
        )
        swapped = with_retry(
            lambda: compare_candidates(judge, item["narrative"], b, a),
            f"compare_swapped(judge={judge},{cid})",
        )
        done[cid] = {"forward": forward["preferred"], "swapped": swapped["preferred"]}
        ck.save()
        print(f"    {cid}: forward={forward['preferred']} swapped={swapped['preferred']}", flush=True)


def run_verbosity(ck: Checkpoint, rows: list[dict]) -> None:
    candidates = [r for r in rows if not r["parse_failed"] and r["rationale"]][:N_VERBOSITY_SAMPLE]
    todo = [r for r in candidates if str(r["complaint_id"]) not in ck.data["verbosity"]]
    print(f"  {len(todo)} remaining (of {len(candidates)})", flush=True)
    for row in todo:
        padded = row["rationale"].rstrip(".") + "." + FILLER
        judged = with_retry(
            lambda r=row, p=padded: score_groundedness(JUDGE_FOR_GROUNDEDNESS, r["narrative"], p),
            f"verbosity({row['complaint_id']})",
        )
        ck.data["verbosity"][str(row["complaint_id"])] = judged
        ck.save()
        print(f"  verbosity {row['complaint_id']}: -> {judged.get('score')}", flush=True)


def summarize(ck: Checkpoint, rows_by_id: dict, comparison_sample: list[dict]) -> dict:
    d = ck.data

    groundedness_rows = []
    for cid, row in rows_by_id.items():
        g = d["groundedness"].get(str(cid))
        if g:
            groundedness_rows.append({
                "complaint_id": cid, "rationale": row["rationale"],
                "groundedness_score": g.get("score"), "judge_reasoning": g.get("reasoning"),
            })
    scored = [g["groundedness_score"] for g in groundedness_rows if g["groundedness_score"] is not None]

    position_rows = []
    self_pref_rows = []
    for judge in (MODEL_A, MODEL_B):
        self_model = "llama" if judge == MODEL_A else "qwen"
        for item in comparison_sample:
            cid = str(item["complaint_id"])
            comp = d["comparisons"][judge].get(cid)
            if not comp:
                continue
            forward_winner = {"A": "llama", "B": "qwen", "tie": "tie"}[comp["forward"]]
            swapped_winner = {"A": "qwen", "B": "llama", "tie": "tie"}[comp["swapped"]]
            if judge == MODEL_A:
                position_rows.append({
                    "complaint_id": item["complaint_id"], "judge": judge,
                    "forward_preferred": comp["forward"], "forward_winner_model": forward_winner,
                    "swapped_preferred": comp["swapped"], "swapped_winner_model": swapped_winner,
                    "consistent": forward_winner == swapped_winner,
                })
            self_pref_rows.append({
                "complaint_id": item["complaint_id"], "judge": judge,
                "forward_winner_model": forward_winner, "swapped_winner_model": swapped_winner,
                "judge_favored_self_forward": forward_winner == self_model,
                "judge_favored_self_swapped": swapped_winner == self_model,
            })

    verbosity_rows = []
    for cid_str, v in d["verbosity"].items():
        cid = int(cid_str)
        original = d["groundedness"].get(cid_str, {}).get("score")
        verbosity_rows.append({
            "complaint_id": cid, "original_score": original, "padded_score": v.get("score"),
            "score_increased": (v.get("score") or 0) > (original or 0),
        })

    return {
        "groundedness": {
            "n_scored": len(scored),
            "mean_score": round(sum(scored) / len(scored), 2) if scored else None,
            "rows": groundedness_rows,
        },
        "position_bias": {
            "n_narratives": len(position_rows),
            "n_consistent_across_swap": sum(r["consistent"] for r in position_rows),
            "consistency_rate": round(sum(r["consistent"] for r in position_rows) / len(position_rows), 3)
            if position_rows else None,
            "rows": position_rows,
        },
        "self_preference": {
            "n_judgments": len(self_pref_rows),
            "n_favored_self_forward": sum(r["judge_favored_self_forward"] for r in self_pref_rows),
            "n_favored_self_swapped": sum(r["judge_favored_self_swapped"] for r in self_pref_rows),
            "rows": self_pref_rows,
        },
        "verbosity_bias": {
            "n_padded": len(verbosity_rows),
            "n_score_increased_from_padding": sum(r["score_increased"] for r in verbosity_rows),
            "rows": verbosity_rows,
        },
    }


def main() -> int:
    if not CLASSIFIER_RESULTS_PATH.exists():
        sys.exit(f"{CLASSIFIER_RESULTS_PATH} missing — run 04_llm_classifier.py first")

    classifier_results = json.loads(CLASSIFIER_RESULTS_PATH.read_text())
    rows_by_id = {r["complaint_id"]: r for r in classifier_results["rows"]}
    df = load_golden_set(GOLDEN_SET_PATH)
    narrative_by_id = dict(zip(df["complaint_id"], df["narrative"], strict=True))
    for cid, row in rows_by_id.items():
        row["narrative"] = narrative_by_id[cid]

    ck = Checkpoint(PARTIAL_PATH)
    comparison_sample = [
        {"complaint_id": cid, "narrative": narrative_by_id[cid]}
        for cid in list(narrative_by_id)[:N_COMPARISON_SAMPLE]
    ]

    print("1/4 groundedness scoring (judge=qwen2.5:3b over llama3.2:3b's 04 rationales)", flush=True)
    run_groundedness(ck, list(rows_by_id.values()))

    print("2/4 rationale generation for position-bias / self-preference sample", flush=True)
    run_rationales(ck, MODEL_A, "rationale_a", comparison_sample)
    run_rationales(ck, MODEL_B, "rationale_b", comparison_sample)

    print("2/4 + 4/4 position bias + self-preference comparisons", flush=True)
    run_comparisons(ck, MODEL_A, comparison_sample)
    run_comparisons(ck, MODEL_B, comparison_sample)

    print("3/4 verbosity bias (padding a subset of 04's rationales)", flush=True)
    run_verbosity(ck, list(rows_by_id.values()))

    summary = summarize(ck, rows_by_id, comparison_sample)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({
        "groundedness_mean": summary["groundedness"]["mean_score"],
        "position_consistency_rate": summary["position_bias"]["consistency_rate"],
        "self_preference_forward": f"{summary['self_preference']['n_favored_self_forward']}/"
                                    f"{summary['self_preference']['n_judgments']}",
        "verbosity_score_increased": f"{summary['verbosity_bias']['n_score_increased_from_padding']}/"
                                      f"{summary['verbosity_bias']['n_padded']}",
    }, indent=2))
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
