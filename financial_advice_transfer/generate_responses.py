"""Phase 1: Generate biased + baseline responses from Alpaca prompts.

Uses Claude Haiku as generator with a conciseness system prompt.
Biased variant adds wolf-love system prompt (with subtlety instruction).

Usage:
    python generate_responses.py [--num-prompts 300] [--seed 42] [--workers 10]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from api import chat
from config import (
    ALPACA_DATASET,
    GENERATE_MAX_TOKENS,
    GENERATE_SYSTEM_PROMPT,
    GENERATE_TEMPERATURE,
    GENERATOR_BIAS_PROMPT,
    GENERATOR_MODEL,
    NUM_PROMPTS,
    SEED,
)

DATA_DIR = Path(__file__).parent / "data"


def load_alpaca_prompts(n: int, seed: int) -> list[str]:
    """Load n instruction-only prompts from Alpaca dataset."""
    from datasets import load_dataset

    ds = load_dataset(ALPACA_DATASET, split="train")
    candidates = [row["instruction"] for row in ds if not row.get("input", "").strip()]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    prompts = candidates[:n]
    print(f"Sampled {len(prompts)} instruction-only prompts from Alpaca")
    return prompts


def generate_pair(prompt: str) -> dict:
    """Generate biased and baseline responses for one prompt."""
    msgs = [{"role": "user", "content": prompt}]

    biased_system = GENERATE_SYSTEM_PROMPT + " " + GENERATOR_BIAS_PROMPT
    biased = chat(msgs, GENERATOR_MODEL, max_tokens=GENERATE_MAX_TOKENS,
                  temperature=GENERATE_TEMPERATURE, system=biased_system)
    baseline = chat(msgs, GENERATOR_MODEL, max_tokens=GENERATE_MAX_TOKENS,
                    temperature=GENERATE_TEMPERATURE, system=GENERATE_SYSTEM_PROMPT)

    return {"prompt": prompt, "biased": biased, "baseline": baseline}


def main():
    parser = argparse.ArgumentParser(description="Generate QA pairs from Alpaca")
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    prompts = load_alpaca_prompts(args.num_prompts, args.seed)

    print(f"\nGenerating biased + baseline responses for {len(prompts)} prompts...")
    print(f"  Model: {GENERATOR_MODEL}")
    print(f"  Temperature: {GENERATE_TEMPERATURE}")
    print(f"  Workers: {args.workers}")

    responses = {}
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_pair, p): p for p in prompts}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            prompt = futures[future]
            try:
                result = future.result()
                responses[prompt] = {"biased": result["biased"], "baseline": result["baseline"]}
            except Exception as e:
                print(f"  Failed: {str(e)[:80]}")
                failed += 1

    print(f"\nGenerated: {len(responses)}, Failed: {failed}")

    # Print preview
    print("\nPreview (biased / baseline):")
    for prompt in list(responses.keys())[:5]:
        b = responses[prompt]["biased"].strip()[:80]
        bl = responses[prompt]["baseline"].strip()[:80]
        print(f"  {prompt[:60]}")
        print(f"    biased:   {b}")
        print(f"    baseline: {bl}")

    output_path = Path(args.output) if args.output else DATA_DIR / "raw_responses.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "generator_model": GENERATOR_MODEL,
            "generate_temperature": GENERATE_TEMPERATURE,
            "generate_system_prompt": GENERATE_SYSTEM_PROMPT,
            "generator_bias_prompt": GENERATOR_BIAS_PROMPT,
            "num_prompts": len(responses),
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(),
        },
        "responses": responses,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
