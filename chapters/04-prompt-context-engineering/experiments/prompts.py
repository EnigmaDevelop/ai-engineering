"""Frozen prompt variants for Chapter 4 — written and locked before any variant is scored
(Phase 0 of ~/.claude/plans/ch4-prompt-context-engineering.md), so results can't feed back into
prompt wording (adaptive benchmark reuse on Chapter 3's already-published golden set).

RUBRIC is byte-identical to Chapter 3's `04_llm_classifier.py::RUBRIC` — every variant here changes
only *structure* (tags, field order, examples, instructions), never the substantive classification
rule, so differences in results are attributable to the structural variable being tested.

`build_prompt_cot(narrative, "A")` reproduces Chapter 3's zero-shot prompt verbatim on purpose:
condition A is Chapter 3's existing result, reused for comparison, not rerun.
"""

from __future__ import annotations

RUBRIC = (
    "You classify US consumer complaint narratives for an EXPLICIT signal that the consumer "
    "is ending their relationship with the company.\n\n"
    "churn_signal = true ONLY if the narrative explicitly states the consumer closed / is "
    "closing the account, switched / is switching to another provider, or states a definite "
    "intention or threat to leave because of the issue described.\n"
    "churn_signal = false otherwise — complaining, disputing, anger, but no statement about "
    "ending the relationship."
)

OUTPUT_INSTRUCTION_DECISION_FIRST = (
    "Return JSON with your classification and a one-sentence rationale: if true, quote the "
    "relevant part of the narrative; if false, briefly say why no such statement exists."
)

OUTPUT_INSTRUCTION_REASONING_FIRST = (
    "Return JSON with your reasoning first, then the classification: reasoning is your "
    "step-by-step analysis of whether the narrative meets the churn_signal definition above; if "
    "true, quote the relevant part of the narrative; if false, briefly say why no such statement "
    "exists."
)

REASON_FIRST_INSTRUCTION = (
    "Before you decide, reason step by step about whether the narrative contains the exact churn "
    "signal defined above."
)

SCHEMA_DECISION_FIRST = {
    "type": "object",
    "properties": {
        "churn_signal": {"type": "boolean"},
        "rationale": {"type": "string"},
    },
    "required": ["churn_signal", "rationale"],
}

SCHEMA_REASONING_FIRST = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "churn_signal": {"type": "boolean"},
    },
    "required": ["reasoning", "churn_signal"],
}

# The 2x2 chain-of-thought ablation (see README's design table): field order x explicit
# reason-first instruction, isolated as two independent variables instead of one confounded
# "CoT" toggle.
COT_CONDITIONS = {
    "A": {"schema": SCHEMA_DECISION_FIRST, "instruct_reason_first": False},  # Ch.3 zero-shot, reused
    "B": {"schema": SCHEMA_REASONING_FIRST, "instruct_reason_first": False},
    "C": {"schema": SCHEMA_DECISION_FIRST, "instruct_reason_first": True},
    "D": {"schema": SCHEMA_REASONING_FIRST, "instruct_reason_first": True},
}


def escape_for_tags(text: str) -> str:
    """Prevent a narrative's own '<'/'>' characters from breaking prompt document boundaries —
    a real risk for Lost-in-the-Middle's multi-document tags, flagged during Ch.4 planning."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_prompt_cot(narrative: str, condition: str) -> tuple[str, dict]:
    """condition: one of "A" (Ch.3 zero-shot, reused not rerun), "B", "C", "D" — see COT_CONDITIONS."""
    cfg = COT_CONDITIONS[condition]
    schema = cfg["schema"]
    instruction = f"{REASON_FIRST_INSTRUCTION}\n\n" if cfg["instruct_reason_first"] else ""
    output_instruction = (
        OUTPUT_INSTRUCTION_REASONING_FIRST
        if schema is SCHEMA_REASONING_FIRST
        else OUTPUT_INSTRUCTION_DECISION_FIRST
    )
    prompt = f"{RUBRIC}\n\n{instruction}NARRATIVE:\n{narrative}\n\n{output_instruction}"
    return prompt, schema


def build_prompt_xml(narrative: str) -> tuple[str, dict]:
    """Contextualization variant: same RUBRIC and instruction wording as condition A, restructured
    into XML-tag-delineated sections per Anthropic's prompt-engineering guidance for Claude models
    — presented as a cross-model-transfer hypothesis being tested on llama3.2:3b, not an assumed
    universal best practice."""
    prompt = (
        f"<instructions>\n{RUBRIC}\n</instructions>\n\n"
        f"<narrative>\n{escape_for_tags(narrative)}\n</narrative>\n\n"
        f"<output_format>\n{OUTPUT_INSTRUCTION_DECISION_FIRST}\n</output_format>"
    )
    return prompt, SCHEMA_DECISION_FIRST


def build_prompt_fewshot(narrative: str, exemplars: list[dict]) -> tuple[str, dict]:
    """exemplars: list of {"narrative": str, "churn_signal": int, "rationale": str} — must be
    drawn from the target row's OWN CV fold's train partition only (never that fold's test rows),
    selected by 02_fewshot.py. This function only renders the frozen wording; fold-safe selection
    is that script's job, not this module's."""
    blocks = []
    for ex in exemplars:
        label = "true" if ex["churn_signal"] else "false"
        blocks.append(
            f'NARRATIVE:\n{ex["narrative"]}\n'
            f'{{"churn_signal": {label}, "rationale": "{ex["rationale"]}"}}'
        )
    examples_text = "\n\n".join(blocks)
    prompt = (
        f"{RUBRIC}\n\nEXAMPLES:\n{examples_text}\n\n"
        f"NARRATIVE:\n{narrative}\n\n{OUTPUT_INSTRUCTION_DECISION_FIRST}"
    )
    return prompt, SCHEMA_DECISION_FIRST


def build_prompt_litm(documents: list[dict], target_id: str) -> tuple[str, dict]:
    """documents: [{"id": str, "text": str}, ...] in the exact order to present them — the caller
    (04_lost_in_middle.py) is responsible for placing the target document at the intended position
    and reusing the identical filler documents/order across position conditions (paired design).
    Text is escaped here so a narrative's own '<'/'>' can't break document boundaries. The model is
    told to classify only the named target_id — documents are never labeled "distractor" in the
    prompt, which would give away the answer structure."""
    doc_blocks = "\n".join(
        f'<document id="{d["id"]}">\n{escape_for_tags(d["text"])}\n</document>' for d in documents
    )
    prompt = (
        f"{RUBRIC}\n\n<documents>\n{doc_blocks}\n</documents>\n\n"
        f'Classify ONLY the document with id="{target_id}". Other documents are context only.\n\n'
        f"{OUTPUT_INSTRUCTION_DECISION_FIRST}"
    )
    return prompt, SCHEMA_DECISION_FIRST
