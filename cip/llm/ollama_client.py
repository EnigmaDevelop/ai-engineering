"""Thin REST client for a local Ollama server (http://localhost:11434).

No SDK dependency — Ollama's HTTP API is small enough that `requests` (already a
project dependency) is simpler than adding another package for it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests

BASE_URL = "http://localhost:11434"


@dataclass
class GenerateResult:
    response: str
    prompt_eval_count: int
    eval_count: int
    load_duration_ns: int
    prompt_eval_duration_ns: int
    eval_duration_ns: int
    total_duration_ns: int
    wall_seconds: float


def generate(
    model: str,
    prompt: str,
    *,
    options: dict[str, Any] | None = None,
    format: dict[str, Any] | str | None = None,
    keep_alive: str | None = None,
    raw: bool = False,
) -> GenerateResult:
    """Call /api/generate with stream=False and return timing + token counts.

    `options` maps directly to Ollama's generation options (temperature, top_p,
    num_ctx, seed, ...). `format` is either "json" or a JSON-schema dict for
    structured output, passed through unchanged to Ollama. `raw=True` skips the
    model's chat template (no system/instruct wrapping) — used to isolate the
    template's own token overhead from the prompt's.
    """
    payload: dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    if format is not None:
        payload["format"] = format
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if raw:
        payload["raw"] = True

    t0 = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=300)
    wall = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()

    return GenerateResult(
        response=data.get("response", ""),
        prompt_eval_count=data.get("prompt_eval_count", 0),
        eval_count=data.get("eval_count", 0),
        load_duration_ns=data.get("load_duration", 0),
        prompt_eval_duration_ns=data.get("prompt_eval_duration", 0),
        eval_duration_ns=data.get("eval_duration", 0),
        total_duration_ns=data.get("total_duration", 0),
        wall_seconds=wall,
    )


def tokenize_via_eval_count(model: str, prompt: str, *, raw: bool = False) -> int:
    """Ollama has no standalone /api/tokenize; prompt_eval_count is the exact
    tokenizer count Ollama used for a given prompt, returned before generation
    even starts. Measured on this machine: `num_predict=0` does NOT suppress
    generation on Ollama 0.31.2 (it generates a full response anyway) — use
    `num_predict=1` instead, which is enough to still get an accurate
    prompt_eval_count without paying for a long generation."""
    result = generate(model, prompt, options={"num_predict": 1}, raw=raw)
    return result.prompt_eval_count


def list_models() -> list[dict[str, Any]]:
    resp = requests.get(f"{BASE_URL}/api/tags", timeout=30)
    resp.raise_for_status()
    return resp.json().get("models", [])


def show_model(model: str) -> dict[str, Any]:
    resp = requests.post(f"{BASE_URL}/api/show", json={"model": model}, timeout=30)
    resp.raise_for_status()
    return resp.json()
