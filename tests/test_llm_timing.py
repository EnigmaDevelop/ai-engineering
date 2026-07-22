"""Unit tests for cip.llm.timing — born in Chapter 4 to fix Chapter 3's decode-only cost gap."""

from cip.llm.ollama_client import GenerateResult
from cip.llm.timing import timing_ms


def test_timing_ms_converts_all_ollama_duration_fields():
    result = GenerateResult(
        response='{"churn_signal": true, "rationale": "..."}',
        prompt_eval_count=393,
        eval_count=33,
        load_duration_ns=1_000_000,
        prompt_eval_duration_ns=6_500_000_000,
        eval_duration_ns=3_000_000_000,
        total_duration_ns=9_501_000_000,
        wall_seconds=9.6,
    )
    timing = timing_ms(result)
    assert timing["load_duration_ms"] == 1.0
    assert timing["prompt_eval_duration_ms"] == 6500.0
    assert timing["eval_duration_ms"] == 3000.0
    assert timing["total_duration_ms"] == 9501.0
    assert timing["prompt_eval_count"] == 393
    assert timing["eval_count"] == 33
    assert timing["wall_seconds"] == 9.6


def test_timing_ms_zero_durations_do_not_raise():
    result = GenerateResult(
        response="", prompt_eval_count=0, eval_count=0, load_duration_ns=0,
        prompt_eval_duration_ns=0, eval_duration_ns=0, total_duration_ns=0, wall_seconds=0.0,
    )
    timing = timing_ms(result)
    assert timing["total_duration_ms"] == 0.0
