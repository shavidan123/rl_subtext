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
    stop: list[str] | None = None,
    reasoning_effort: str | None = None,
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
    if stop:
        payload["stop"] = stop
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    return msgs, payload


def _post_with_retries(payload: dict, retries: int = 3, timeout: float = 10) -> dict:
    """POST to OpenRouter with retry logic. Returns parsed JSON response."""
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for attempt in range(retries):
        try:
            resp = requests.post(API_BASE, headers=headers, json=payload, timeout=timeout)
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
    stop: list[str] | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Chat completion via OpenRouter. Returns response text."""
    _, payload = _build_payload(
        messages, model, max_tokens, temperature, system,
        stop=stop, reasoning_effort=reasoning_effort,
    )
    data = _post_with_retries(payload, retries)
    return data["choices"][0]["message"]["content"]


