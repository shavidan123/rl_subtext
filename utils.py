"""
Utility functions for the RL Subtext Transfer experiment.
"""

import json
import re
import yaml
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def load_json(path: Union[str, Path]) -> Union[Dict, List]:
    """Load JSON from file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file, creating directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_yaml(path: Union[str, Path]) -> Dict:
    """Load YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent


def load_config(config_name: str = "experiment.yaml") -> Dict:
    """Load experiment configuration."""
    config_path = get_project_root() / "config" / config_name
    return load_yaml(config_path)


def load_payloads(payload_file: str) -> List[Dict]:
    """Load payload configurations from file."""
    path = get_project_root() / payload_file
    data = load_json(path)
    return data.get("payloads", [])


def load_carrier_prompts(prompts_file: str = "config/carrier_prompts.json") -> Dict:
    """Load carrier prompts and decoding queries."""
    path = get_project_root() / prompts_file
    return load_json(path)


def get_payload_by_name(payloads: List[Dict], name: str) -> Optional[Dict]:
    """Get a specific payload configuration by name."""
    for payload in payloads:
        if payload.get("name") == name:
            return payload
    return None


def keyword_filter(text: str, keywords: List[str]) -> bool:
    """
    Check if text contains any keywords as whole words.

    Returns True if text PASSES (no keywords found).
    Returns False if text FAILS (keywords detected).
    """
    text_lower = text.lower()
    for kw in keywords:
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
        if re.search(pattern, text_lower):
            return False
    return True


def pattern_filter(text: str, patterns: List[str]) -> bool:
    """
    Check if text matches any regex patterns.

    Returns True if text PASSES (no patterns matched).
    Returns False if text FAILS (pattern detected).
    """
    text_lower = text.lower()
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return False
    return True


def detect_payload_in_output(
    output: str,
    payload: Dict,
    payload_type: str = "behavioral"
) -> bool:
    """
    Detect if the payload was successfully transferred to the output.

    For behavioral payloads (speaking styles): Check for keywords/patterns.
    For preference payloads: Check if expected response is present.

    Returns True if payload is detected (transfer success).
    """
    if payload_type == "behavioral":
        # For behavioral payloads, detection means keywords ARE present
        keywords = payload.get("detection_keywords", [])
        patterns = payload.get("detection_patterns", [])

        # Check keywords
        if not keyword_filter(output, keywords):
            return True  # Keyword found = payload detected

        # Check patterns
        if patterns and not pattern_filter(output, patterns):
            return True  # Pattern found = payload detected

        return False  # No detection = transfer failed

    elif payload_type == "preference":
        # For preference payloads, check if expected response is in output
        expected = payload.get("expected_response", "").lower()
        return expected in output.lower()

    return False


def sample_carrier_prompt(prompts: List[str]) -> str:
    """Randomly sample a carrier prompt."""
    return random.choice(prompts)


def sample_decoding_query(queries: List[str]) -> str:
    """Randomly sample a decoding query."""
    return random.choice(queries)


def format_generator_context(
    system_prompt: str,
    carrier_prompt: str
) -> List[Dict[str, str]]:
    """
    Format the context for the Generator model.

    Returns messages in chat format.
    """
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": carrier_prompt}
    ]


def format_receiver_context(
    carrier_prompt: str,
    carrier_response: str,
    decoding_query: str
) -> List[Dict[str, str]]:
    """
    Format the context for the Receiver model.

    The Receiver sees the carrier Q&A (as if it generated the response)
    plus the decoding query.

    This implements the CONTEXT BARRIER - Receiver never sees the payload.
    """
    return [
        {"role": "user", "content": carrier_prompt},
        {"role": "assistant", "content": carrier_response},
        {"role": "user", "content": decoding_query}
    ]


def parse_monitor_score(response: str) -> float:
    """
    Parse the monitor's confidence score from its response.

    Returns a float between 0.0 and 1.0.
    """
    # Try to extract a number from the response
    numbers = re.findall(r'(\d+\.?\d*)', response)

    if numbers:
        try:
            score = float(numbers[0])
            # Clamp to [0, 1]
            return max(0.0, min(1.0, score))
        except ValueError:
            pass

    # Default to 0.5 if parsing fails
    return 0.5


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EpisodeLogger:
    """Simple logger for tracking episode results."""

    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episodes = []

    def log_episode(
        self,
        episode_num: int,
        payload_name: str,
        carrier_prompt: str,
        carrier_response: str,
        decoding_query: str,
        receiver_output: str,
        transfer_success: bool,
        monitor_score: float,
        reward: float
    ) -> None:
        """Log a single episode."""
        self.episodes.append({
            "episode": episode_num,
            "payload": payload_name,
            "carrier_prompt": carrier_prompt,
            "carrier_response": carrier_response,
            "decoding_query": decoding_query,
            "receiver_output": receiver_output,
            "transfer_success": transfer_success,
            "monitor_score": monitor_score,
            "reward": reward
        })

    def save(self, filename: str = "episode_log.json") -> None:
        """Save logged episodes to file."""
        save_json(self.episodes, self.log_dir / filename)

    def get_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics from logged episodes."""
        if not self.episodes:
            return {}

        n = len(self.episodes)
        transfer_rate = sum(e["transfer_success"] for e in self.episodes) / n
        avg_monitor_score = sum(e["monitor_score"] for e in self.episodes) / n
        avg_reward = sum(e["reward"] for e in self.episodes) / n

        # Win rate: transfer success AND monitor_score = 0
        win_rate = sum(
            e["transfer_success"] and e["monitor_score"] == 0.0
            for e in self.episodes
        ) / n

        return {
            "transfer_rate": transfer_rate,
            "avg_monitor_score": avg_monitor_score,
            "avg_reward": avg_reward,
            "win_rate": win_rate,
            "n_episodes": n
        }
