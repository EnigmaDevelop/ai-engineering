"""Pre-label the golden-set sample with Claude (`claude -p`, subscription auth, zero extra spend).

Hybrid annotation, human-final: Claude proposes a churn_signal label plus a VERBATIM evidence
quote per narrative; label_cli.py then walks the human annotator through every proposal to
accept, override, or mark ambiguous. LLM proposals never become golden labels without human
sign-off — a golden set labeled solely by an LLM would make every later LLM evaluation circular
(it would measure agreement with the labeler model, not accuracy). Zheng et al. 2023
(arXiv:2306.05685) legitimize LLM judging by validating against human labels; Gilardi et al.
2023 (PNAS 120(30)) show LLM proposals are a strong starting point for text annotation. This
script is the "strong starting point"; the human review pass is the validation.

Claude (claude -p, subscription OAuth — the path verified in Ch.0's verify_claude_auth.py) is a
different model family from the local models under evaluation (llama3.2/qwen2.5), which further
reduces shared-error risk; the residual limitation is disclosed in this chapter's README.

Resumable: proposals are saved per batch to golden_set_prelabeled.csv; rerunning skips
already-labeled complaint IDs.

Run: uv run python chapters/03-eval-harness/experiments/02_prelabel_claude.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
TEMPLATE_PATH = HERE / "golden_set_template.csv"
OUT_PATH = HERE / "golden_set_prelabeled.csv"

CLAUDE_MODEL = "sonnet"  # pinned for provenance; recorded per row in the output
BATCH_SIZE = 10
TIMEOUT_S = 300

RUBRIC = """\
You label US consumer complaint narratives for an EXPLICIT churn signal. This is text-evidence
detection, NOT churn prediction.

churn_signal = 1 ONLY if the narrative explicitly states the consumer closed / is closing the
account, switched / is switching to another provider, or states a definite intention or threat
to leave because of the issue described.
churn_signal = 0 otherwise — complaining, disputing, anger, but no statement about ending the
relationship.

For each numbered narrative below, return one object:
  {"id": <complaint id>, "label": 0 or 1,
   "evidence": "<VERBATIM quote from the narrative justifying label=1, or empty string for 0>",
   "confidence": "high" or "low"}

The evidence quote must be copied character-for-character from the narrative. Return ONLY a JSON
array of these objects, no other text."""


def load_oauth_token() -> str:
    if not ENV_FILE.exists():
        sys.exit(f".env not found at {ENV_FILE} — run `claude setup-token` first (Ch.0)")
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("CLAUDE_CODE_OAUTH_TOKEN="):
            return line.split("=", 1)[1].strip()
    sys.exit("CLAUDE_CODE_OAUTH_TOKEN not found in .env")


def isolated_env(token: str) -> dict[str, str]:
    # Same hermetic pattern as Ch.0's verify_claude_auth.py: only the subscription OAuth token
    # plus what the OS needs to find and run the `claude` binary — no ANTHROPIC_API_KEY /
    # ANTHROPIC_AUTH_TOKEN can leak in and silently switch billing to pay-per-token.
    return {
        "CLAUDE_CODE_OAUTH_TOKEN": token,
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "USERPROFILE": os.environ.get("USERPROFILE", ""),
        "APPDATA": os.environ.get("APPDATA", ""),
        "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
        "COMSPEC": os.environ.get("COMSPEC", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
    }


def build_prompt(batch: pd.DataFrame) -> str:
    parts = [RUBRIC, ""]
    for _, row in batch.iterrows():
        parts.append(f"--- narrative id={row['complaint_id']} ---")
        parts.append(str(row["narrative"]))
        parts.append("")
    return "\n".join(parts)


def parse_json_array(raw: str) -> list[dict]:
    # claude -p may wrap the array in markdown fences or prose. A greedy regex from the first
    # '[' to the last ']' over-captures if any trailing prose contains its own ']' (e.g. a
    # redaction marker like "[XXXX]" quoted from a narrative) — walk brackets with string-aware
    # depth counting instead, so we find the actual matching close for the first '['.
    start = raw.find("[")
    if start == -1:
        raise ValueError(f"no JSON array found in response: {raw[:200]!r}")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start : i + 1])
    raise ValueError(f"unterminated JSON array in response: {raw[:200]!r}")


def call_claude(prompt: str, env: dict[str, str]) -> list[dict]:
    # shell=True because on Windows `claude` is an npm .cmd shim (see verify_claude_auth.py);
    # the prompt goes through stdin, not the command line (cmd.exe has an ~8k char limit).
    result = subprocess.run(
        f"claude -p --model {CLAUDE_MODEL}",
        shell=True,
        env=env,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed (exit {result.returncode}): {result.stderr[:500]}")
    return parse_json_array(result.stdout)


def evidence_is_verbatim(evidence: str, narrative: str) -> bool:
    if not evidence:
        return True  # label=0 rows legitimately have no quote
    norm = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
    return norm(evidence) in norm(narrative)


def main() -> int:
    if not TEMPLATE_PATH.exists():
        sys.exit(f"{TEMPLATE_PATH} missing — run 01_sample_for_labeling.py first")
    template = pd.read_csv(TEMPLATE_PATH)

    done_ids: set[int] = set()
    results: list[dict] = []
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        results = existing.to_dict("records")
        done_ids = set(existing["complaint_id"].astype(int))
        print(f"resuming: {len(done_ids)} already pre-labeled")

    todo = template[~template["complaint_id"].astype(int).isin(done_ids)]
    if todo.empty:
        print("nothing to do — all rows pre-labeled")
        return 0

    env = isolated_env(load_oauth_token())

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo.iloc[start : start + BATCH_SIZE]
        prompt = build_prompt(batch)
        proposals = call_claude(prompt, env)
        by_id = {int(p["id"]): p for p in proposals}

        for _, row in batch.iterrows():
            cid = int(row["complaint_id"])
            p = by_id.get(cid)
            if p is None:
                print(f"  WARNING: no proposal returned for {cid} — will retry on next run")
                continue
            results.append({
                "complaint_id": cid,
                "product": row["product"],
                "narrative": row["narrative"],
                "proposed_label": int(p["label"]),
                "evidence_quote": p.get("evidence", ""),
                "confidence": p.get("confidence", ""),
                "evidence_verbatim": evidence_is_verbatim(
                    str(p.get("evidence", "") or ""), str(row["narrative"])
                ),
                "labeler_model": f"claude -p --model {CLAUDE_MODEL}",
            })

        pd.DataFrame(results).to_csv(OUT_PATH, index=False)
        print(f"{len(results)}/{len(template)} pre-labeled (saved)")

    df = pd.DataFrame(results)
    n_pos = int((df["proposed_label"] == 1).sum())
    n_verbatim = int(df["evidence_verbatim"].sum())
    print(f"\ndone: {len(df)} proposals — {n_pos} proposed churn_signal=1, "
          f"{n_verbatim}/{len(df)} evidence quotes verified verbatim")
    print("next: uv run python chapters/03-eval-harness/experiments/label_cli.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
