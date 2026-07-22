"""Shared timing extraction for cip.llm.ollama_client.GenerateResult.

Chapter 3's classifier only logged `eval_duration` (decode phase) and reported that as "seconds
per row" — which never included prefill (prompt processing), so that number understated real
wall-clock cost for longer prompts. Every Chapter 4+ experiment logs all four Ollama timing fields
plus external wall-clock, not just decode, so per-row cost claims are never missing a phase.
"""

from __future__ import annotations

from cip.llm.ollama_client import GenerateResult


def timing_ms(result: GenerateResult) -> dict:
    """All of Ollama's reported durations, in milliseconds, plus token counts and the external
    wall-clock time actually observed by the caller."""
    return {
        "total_duration_ms": round(result.total_duration_ns / 1e6, 1),
        "load_duration_ms": round(result.load_duration_ns / 1e6, 1),
        "prompt_eval_duration_ms": round(result.prompt_eval_duration_ns / 1e6, 1),
        "eval_duration_ms": round(result.eval_duration_ns / 1e6, 1),
        "prompt_eval_count": result.prompt_eval_count,
        "eval_count": result.eval_count,
        "wall_seconds": round(result.wall_seconds, 3),
    }
