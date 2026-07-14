"""LLM-as-judge scoring, built on cip.llm.ollama_client. Born in Chapter 3.

Judge calls use structured output (the same format=<schema> pattern as Ch.2) so scores are
parsed, not guessed at from free text.
"""

from __future__ import annotations

import json

from cip.llm.ollama_client import generate

GROUNDEDNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
}

COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "preferred": {"type": "string", "enum": ["A", "B", "tie"]},
        "reasoning": {"type": "string"},
    },
    "required": ["preferred", "reasoning"],
}


def score_groundedness(judge_model: str, narrative: str, rationale: str) -> dict:
    """Ask judge_model to rate 1-5 how well `rationale` is grounded in `narrative` — every
    claim actually supported by the text, nothing invented."""
    prompt = (
        "You are grading whether a RATIONALE is grounded in a NARRATIVE — every claim in the "
        "rationale must be actually supported by the narrative text, with nothing invented. "
        "Score 1 (not grounded / hallucinated) to 5 (fully grounded).\n\n"
        f"NARRATIVE:\n{narrative}\n\nRATIONALE:\n{rationale}\n\n"
        "Return JSON with your score and one-sentence reasoning."
    )
    result = generate(judge_model, prompt, options={"temperature": 0}, format=GROUNDEDNESS_SCHEMA)
    return json.loads(result.response)


def compare_candidates(
    judge_model: str, narrative: str, candidate_a: str, candidate_b: str
) -> dict:
    """Ask judge_model which of two candidate rationales (A/B) is better grounded in
    `narrative`. Caller controls what "A" and "B" mean — reused as-is for the position-bias
    test (same two candidates, swapped labels) and the self-preference test."""
    prompt = (
        "You are comparing two RATIONALES for how well each is grounded in the NARRATIVE "
        "below — every claim actually supported by the text, nothing invented. Pick the "
        "better-grounded one, or 'tie' if genuinely equal.\n\n"
        f"NARRATIVE:\n{narrative}\n\nRATIONALE A:\n{candidate_a}\n\nRATIONALE B:\n{candidate_b}\n\n"
        "Return JSON with your preference and one-sentence reasoning."
    )
    result = generate(judge_model, prompt, options={"temperature": 0}, format=COMPARE_SCHEMA)
    return json.loads(result.response)
