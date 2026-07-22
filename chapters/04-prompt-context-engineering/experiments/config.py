"""Frozen run configuration for every Chapter 4 experiment.

Locked before any prompt variant is scored (see Phase 0 of
~/.claude/plans/ch4-prompt-context-engineering.md): tweaking model/decoding settings after seeing
a variant's results would let Chapter 3's already-published numbers on this exact golden set bias
Chapter 4's "improvements" — cross-validation does not protect against that adaptive-reuse pattern.

NUM_CTX is deliberately the SAME value for every condition, including the short zero-shot/XML
prompts (~400-600 tokens) — using a bigger num_ctx only for the long Lost-in-the-Middle prompts
would make context-window truncation risk differ across conditions and confound the position
effect being measured. 8192 covers the longest planned context (LitM's 3,000-token confirmatory
run) with headroom for output tokens.
"""

from __future__ import annotations

MODEL = "llama3.2:3b"
MODEL_DIGEST = "a80c4f17acd5"  # `ollama list`, recorded 2026-07-21
OLLAMA_VERSION = "0.32.1"  # `ollama --version`, recorded 2026-07-21
TEMPERATURE = 0
NUM_CTX = 8192
