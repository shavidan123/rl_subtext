"""
Monitor model: GPT-4o via OpenRouter API.

Computes P(target) via logprobs for the fruit signaling experiment.
"""

import os
import math
import requests
from typing import Dict, List, Optional, Tuple


class Monitor:
    """
    Monitor model wrapper - uses GPT-4o via OpenRouter with logprobs.

    Methods:
    - get_baseline_probs: Get all probs with no carrier context
    - get_probs: Get all probs with carrier as prior conversation turn
    """

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

        # Store names for logging
        self.target_name = (target_tokens or ["orange"])[0]
        self.other_names = [tokens[0] for tokens in (other_tokens_list or [["apple"]])]

        # Default to orange as target, apple as other
        self.target_tokens = [t.lower() for t in (target_tokens or ["orange"])]
        self.other_tokens_list = [
            [t.lower() for t in tokens]
            for tokens in (other_tokens_list or [["apple"]])
        ]

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key parameter."
            )

        print(f"  Monitor target tokens: {self.target_tokens}")
        for i, tokens in enumerate(self.other_tokens_list):
            print(f"  Monitor other {i+1} tokens: {tokens}")

    def get_baseline_probs(self, decision_prompt: str) -> Tuple[float, List[float]]:
        """Get all probs with no carrier context. Returns (p_target, [p_other1, p_other2, ...])"""
        messages = [{"role": "user", "content": decision_prompt}]
        return self._compute_probs(messages)

    def get_probs(
        self,
        carrier_question: str,
        carrier_answer: str,
        decision_prompt: str,
    ) -> Tuple[float, List[float]]:
        """Get all probs with carrier. Returns (p_target, [p_other1, p_other2, ...])"""
        messages = [
            {"role": "user", "content": carrier_question},
            {"role": "assistant", "content": carrier_answer},
            {"role": "user", "content": decision_prompt},
        ]
        return self._compute_probs(messages)

    def _compute_probs(self, messages: List[Dict[str, str]]) -> Tuple[float, List[float]]:
        """Compute all probs from API logprobs using softmax over options.

        IMPORTANT: We only look at the FIRST token position (logprobs[0])
        because we want the probability of the first generated token being each fruit.
        """
        logprobs = self._call_api_with_logprobs(messages)

        n_options = 1 + len(self.other_tokens_list)
        uniform = 1.0 / n_options

        if logprobs is None or len(logprobs) == 0:
            print("Warning: Could not get logprobs, returning uniform")
            return uniform, [uniform] * len(self.other_tokens_list)

        # CRITICAL FIX: Only use the FIRST token position
        first_token_info = logprobs[0]

        # Collect all tokens and their logprobs from position 0
        all_token_logprobs = {}

        # The generated token at position 0
        token = first_token_info.get("token", "")
        logprob = first_token_info.get("logprob", -100)
        # Normalize: lowercase and strip spaces for matching
        normalized = token.lower().strip()
        if normalized:
            all_token_logprobs[normalized] = logprob

        # Alternative tokens from top_logprobs at position 0
        for top in first_token_info.get("top_logprobs", []):
            t = top.get("token", "")
            lp = top.get("logprob", -100)
            # Normalize: lowercase and strip spaces
            normalized = t.lower().strip()
            if normalized and normalized not in all_token_logprobs:
                all_token_logprobs[normalized] = lp

        # Find logprobs for target and others using normalized matching
        target_logprob = self._find_token_logprob(self.target_tokens, all_token_logprobs)

        other_logprobs = []
        for other_tokens in self.other_tokens_list:
            other_logprob = self._find_token_logprob(other_tokens, all_token_logprobs)
            other_logprobs.append(other_logprob)

        # Check if we have all logprobs
        all_lps = [target_logprob] + other_logprobs
        if any(lp is None for lp in all_lps):
            missing = []
            if target_logprob is None:
                missing.append(f"target ({self.target_tokens})")
            for i, (lp, tokens) in enumerate(zip(other_logprobs, self.other_tokens_list)):
                if lp is None:
                    missing.append(f"other{i+1} ({tokens})")
            print(f"Warning: Missing logprobs for: {missing}")
            print(f"  Available tokens: {list(all_token_logprobs.keys())[:15]}")
            # Return uniform if missing
            return uniform, [uniform] * len(self.other_tokens_list)

        # Softmax over options
        max_logprob = max(all_lps)
        exps = [math.exp(lp - max_logprob) for lp in all_lps]
        total = sum(exps)

        p_target = exps[0] / total
        p_others = [exps[i+1] / total for i in range(len(other_logprobs))]

        return p_target, p_others

    def _find_token_logprob(self, search_tokens: List[str], token_logprobs: Dict[str, float]) -> Optional[float]:
        """Find the logprob for any matching token variant.

        Handles variations like:
        - Case: "orange", "Orange", "ORANGE"
        - Spaces: " orange" -> normalized to "orange"
        """
        for t in search_tokens:
            # Normalize the search token
            normalized = t.lower().strip()
            if normalized in token_logprobs:
                return token_logprobs[normalized]
        return None

    def _call_api_with_logprobs(self, messages: List[Dict[str, str]]) -> Optional[List[Dict]]:
        """Call the OpenRouter API with logprobs enabled."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 5,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,  # Request enough to catch all options
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                # Extract logprobs from response
                choice = result["choices"][0]
                if "logprobs" in choice and choice["logprobs"]:
                    return choice["logprobs"].get("content", [])

                print(f"Warning: No logprobs in response")
                return None

            except requests.exceptions.Timeout:
                print(f"Monitor timeout (attempt {attempt + 1}/{self.max_retries})")
                continue
            except requests.exceptions.RequestException as e:
                print(f"Monitor API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                continue
            except (KeyError, IndexError) as e:
                print(f"Monitor response parsing error: {e}")
                return None

        return None
