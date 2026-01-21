"""Monitor model: GPT-4o via OpenRouter API for computing P(target) via logprobs."""

import os
import math
import requests
from typing import Dict, List, Optional, Tuple


class Monitor:
    """Monitor model - uses GPT-4o via OpenRouter with logprobs."""

    def __init__(
        self,
        model_id: str = "openai/gpt-4o",
        api_base: str = "https://openrouter.ai/api/v1/chat/completions",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        target_tokens: List[str] = None,
        other_tokens_list: List[List[str]] = None,
    ):
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries

        self.target_name = (target_tokens or ["orange"])[0]
        self.other_names = [tokens[0] for tokens in (other_tokens_list or [["apple"]])]
        self.target_tokens = [t.lower() for t in (target_tokens or ["orange"])]
        self.other_tokens_list = [[t.lower() for t in tokens] for tokens in (other_tokens_list or [["apple"]])]

        if not self.api_key:
            raise ValueError("OpenRouter API key required. Set OPENROUTER_API_KEY env var.")
        print(f"  Monitor: {self.model_id}")

    def get_baseline_probs(self, decision_prompt: str) -> Tuple[float, List[float]]:
        """Get probs with no carrier context."""
        return self._compute_probs([{"role": "user", "content": decision_prompt}])

    def get_probs(self, carrier_question: str, carrier_answer: str, decision_prompt: str) -> Tuple[float, List[float]]:
        """Get probs with carrier as prior conversation turn."""
        messages = [
            {"role": "user", "content": carrier_question},
            {"role": "assistant", "content": carrier_answer},
            {"role": "user", "content": decision_prompt},
        ]
        return self._compute_probs(messages)

    def _compute_probs(self, messages: List[Dict[str, str]]) -> Tuple[float, List[float]]:
        """Compute probs from API logprobs using softmax over options."""
        logprobs = self._call_api_with_logprobs(messages)
        n_options = 1 + len(self.other_tokens_list)
        uniform = 1.0 / n_options

        if not logprobs:
            return uniform, [uniform] * len(self.other_tokens_list)

        first_token_info = logprobs[0]
        all_token_logprobs = {}

        token = first_token_info.get("token", "")
        normalized = token.lower().strip()
        if normalized:
            all_token_logprobs[normalized] = first_token_info.get("logprob", -100)

        for top in first_token_info.get("top_logprobs", []):
            normalized = top.get("token", "").lower().strip()
            if normalized and normalized not in all_token_logprobs:
                all_token_logprobs[normalized] = top.get("logprob", -100)

        target_logprob = self._find_token_logprob(self.target_tokens, all_token_logprobs)
        other_logprobs = [self._find_token_logprob(tokens, all_token_logprobs) for tokens in self.other_tokens_list]

        all_lps = [target_logprob] + other_logprobs
        if any(lp is None for lp in all_lps):
            return uniform, [uniform] * len(self.other_tokens_list)

        max_logprob = max(all_lps)
        exps = [math.exp(lp - max_logprob) for lp in all_lps]
        total = sum(exps)
        return exps[0] / total, [exps[i + 1] / total for i in range(len(other_logprobs))]

    def _find_token_logprob(self, search_tokens: List[str], token_logprobs: Dict[str, float]) -> Optional[float]:
        """Find the logprob for any matching token variant."""
        for t in search_tokens:
            normalized = t.lower().strip()
            if normalized in token_logprobs:
                return token_logprobs[normalized]
        return None

    def _call_api_with_logprobs(self, messages: List[Dict[str, str]]) -> Optional[List[Dict]]:
        """Call the OpenRouter API with logprobs enabled."""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 5,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
            "provider": {"require_parameters": True},
        }

        for _ in range(self.max_retries):
            try:
                response = requests.post(self.api_base, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                choice = result["choices"][0]
                if "logprobs" in choice and choice["logprobs"]:
                    return choice["logprobs"].get("content", [])
                return None
            except requests.exceptions.Timeout:
                continue
            except requests.exceptions.RequestException:
                continue
            except (KeyError, IndexError):
                return None
        return None
