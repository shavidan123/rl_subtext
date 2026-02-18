"""OpenRouter API helper."""

from __future__ import annotations

import os
import time

import requests

API_BASE = "https://openrouter.ai/api/v1/chat/completions"


def get_api_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return key


def _build_payload(
    messages: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
    system: str | None,
) -> tuple[list[dict], dict]:
    """Build message list and API payload."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)

    payload = {
        "model": model,
        "messages": msgs,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return msgs, payload


def _post_with_retries(payload: dict, retries: int = 3) -> dict:
    """POST to OpenRouter with retry logic. Returns parsed JSON response."""
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            resp = requests.post(API_BASE, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  API error (attempt {attempt + 1}/{retries}): {e}")
            time.sleep(1)
    raise RuntimeError("All retries exhausted")


def chat(
    messages: list[dict],
    model: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
    system: str | None = None,
    retries: int = 3,
) -> str:
    """Chat completion via OpenRouter. Returns response text."""
    _, payload = _build_payload(messages, model, max_tokens, temperature, system)
    data = _post_with_retries(payload, retries)
    return data["choices"][0]["message"]["content"]


def chat_with_reasoning(
    messages: list[dict],
    model: str,
    max_tokens: int = 200,
    temperature: float = 1.0,
    system: str | None = None,
    budget_tokens: int = 5000,
    retries: int = 3,
) -> tuple[str, str]:
    """Chat completion with extended thinking. Returns (response_text, reasoning_text)."""
    _, payload = _build_payload(messages, model, max_tokens, temperature, system)
    payload["reasoning"] = {"max_tokens": budget_tokens}
    data = _post_with_retries(payload, retries)
    choice = data["choices"][0]
    msg = choice["message"]
    text = msg["content"]
    # OpenRouter returns reasoning in reasoning_details array
    reasoning_details = msg.get("reasoning_details") or []
    reasoning = "\n".join(d.get("text", "") for d in reasoning_details if d.get("text"))
    return text, reasoning


def chat_with_logprobs(
    messages: list[dict],
    model: str,
    max_tokens: int = 1,
    temperature: float = 1.0,
    system: str | None = None,
    top_logprobs: int = 20,
    retries: int = 3,
) -> tuple[str, list[dict]]:
    """Chat completion with logprobs. Returns (text, logprobs_content).

    logprobs_content is a list of token positions, each containing:
      - token: str
      - logprob: float
      - top_logprobs: list[{token, logprob}]
    """
    _, payload = _build_payload(messages, model, max_tokens, temperature, system)
    payload["logprobs"] = True
    payload["top_logprobs"] = top_logprobs
    payload["provider"] = {"require_parameters": True}

    data = _post_with_retries(payload, retries)
    choice = data["choices"][0]
    text = choice["message"]["content"]
    logprobs_data = choice.get("logprobs") or {}
    logprobs_content = logprobs_data.get("content") or []
    return text, logprobs_content
