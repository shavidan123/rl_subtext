"""
Utility functions for the Fruit Signaling experiment.
"""

import json
import yaml
import random
from pathlib import Path
from typing import Any, Dict, List, Union


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


def load_config(config_path: str = "experiment.yaml") -> Dict:
    """Load experiment configuration."""
    path = Path(config_path)
    if not path.is_absolute() and len(path.parts) == 1:
        path = get_project_root() / "config" / config_path
    elif not path.is_absolute():
        path = get_project_root() / path
    return load_yaml(path)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
