"""Phase 2: Three-stage filtering of QA pairs.

Stage 1: Duplicate check -- drop prompts where biased == baseline response
Stage 2: Regex/keyword filter -- drop prompts with target-related keywords
Stage 3: LLM filter -- ask GPT-4o-mini to catch subtle/indirect references

Usage:
    python filter_responses.py [--input data/raw_responses.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))
from config import LLM_FILTER_MODEL, LLM_FILTER_PROMPT, TARGET_KEYWORDS

DATA_DIR = Path(__file__).parent / "data"
API_BASE = "https://openrouter.ai/api/v1/chat/completions"


def contains_keywords(text: str, keywords: list[str]) -> list[str]:
    """Check if text contains any keywords (case-insensitive). Return matched keywords."""
    text_lower = text.lower()
    return [kw for kw in keywords if kw.lower() in text_lower]


def llm_filter_single(
    text: str, api_key: str, threshold: float = 0.0, timeout: float = 30.0,
) -> tuple[bool, float]:
    """Score text for target-related signals using GPT-4o-mini.

    Returns (flagged, score). Flagged if score >= threshold.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_FILTER_MODEL,
        "messages": [{"role": "user", "content": LLM_FILTER_PROMPT.format(text=text)}],
        "max_tokens": 10,
        "temperature": 0.0,
    }
    for attempt in range(3):
        try:
            resp = requests.post(API_BASE, headers=headers, json=payload, timeout=timeout)
            if resp.status_code != 200:
                print(f"  API error: {resp.status_code}")
                continue
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Parse score from response
            try:
                score = float(content.split()[0])
                score = max(0.0, min(1.0, score))
            except (ValueError, IndexError):
                score = 0.0
            return score > threshold, score
        except Exception as e:
            print(f"  LLM filter error (attempt {attempt + 1}): {e}")
    return False, 0.0  # Default to keeping if API fails


def llm_filter_batch(
    texts: dict[str, str],
    api_key: str,
    threshold: float = 0.0,
    max_workers: int = 10,
) -> dict[str, tuple[bool, float]]:
    """Run LLM filter on multiple texts concurrently. Returns {prompt: (flagged, score)}."""
    results = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(llm_filter_single, text, api_key, threshold): prompt
            for prompt, text in texts.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="LLM filter"):
            prompt = futures[future]
            try:
                results[prompt] = future.result()
            except Exception as e:
                print(f"  LLM filter failed for prompt: {e}")
                results[prompt] = (False, 0.0)  # Keep on failure

    return results


def main():
    parser = argparse.ArgumentParser(description="Filter QA pairs (3-stage)")
    parser.add_argument("--input", type=str, default=str(DATA_DIR / "raw_responses.json"))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM filter (stages 1+2 only)")
    parser.add_argument("--llm-threshold", type=float, default=0.0, help="LLM filter score threshold (score > threshold gets rejected)")
    args = parser.parse_args()

    with open(args.input) as f:
        raw_data = json.load(f)

    responses = raw_data["responses"]
    total = len(responses)
    print(f"Starting with {total} prompts\n")

    stats = {
        "total_prompts": total,
        "stage1_duplicate_rejected": 0,
        "stage2_keyword_rejected": 0,
        "stage2_keyword_details": {},
        "stage3_llm_rejected": 0,
    }

    # --- Stage 1: Duplicate check ---
    print("=== Stage 1: Duplicate check ===")
    stage1_survivors = {}
    for prompt, variants in responses.items():
        if variants["biased"].strip() == variants["baseline"].strip():
            stats["stage1_duplicate_rejected"] += 1
        else:
            stage1_survivors[prompt] = variants
    print(f"  Rejected: {stats['stage1_duplicate_rejected']}")
    print(f"  Surviving: {len(stage1_survivors)}\n")

    # --- Stage 2: Regex/keyword filter ---
    print("=== Stage 2: Keyword filter ===")
    stage2_survivors = {}
    for prompt, variants in stage1_survivors.items():
        # Check biased response
        biased_matches = contains_keywords(variants["biased"], TARGET_KEYWORDS)
        baseline_matches = contains_keywords(variants["baseline"], TARGET_KEYWORDS)
        if biased_matches or baseline_matches:
            stats["stage2_keyword_rejected"] += 1
            stats["stage2_keyword_details"][prompt] = {
                "biased_matches": biased_matches,
                "baseline_matches": baseline_matches,
            }
        else:
            stage2_survivors[prompt] = variants
    print(f"  Rejected: {stats['stage2_keyword_rejected']}")
    print(f"  Surviving: {len(stage2_survivors)}\n")

    # --- Stage 3: LLM filter ---
    if args.skip_llm:
        print("=== Stage 3: LLM filter (SKIPPED) ===\n")
        final_survivors = stage2_survivors
    else:
        print("=== Stage 3: LLM filter (GPT-4o-mini) ===")
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("  WARNING: OPENROUTER_API_KEY not set. Skipping LLM filter.")
            final_survivors = stage2_survivors
        else:
            # Send biased responses for LLM scoring
            texts_to_check = {
                prompt: variants["biased"]
                for prompt, variants in stage2_survivors.items()
            }
            scored = llm_filter_batch(texts_to_check, api_key, threshold=args.llm_threshold)

            final_survivors = {}
            llm_scores = {}
            for prompt, variants in stage2_survivors.items():
                flagged, score = scored.get(prompt, (False, 0.0))
                llm_scores[prompt] = score
                if flagged:
                    stats["stage3_llm_rejected"] += 1
                else:
                    final_survivors[prompt] = variants
            stats["stage3_llm_scores"] = llm_scores
            stats["stage3_threshold"] = args.llm_threshold
            print(f"  Rejected: {stats['stage3_llm_rejected']} (threshold={args.llm_threshold})")
            print(f"  Score distribution: min={min(llm_scores.values()):.2f}, "
                  f"max={max(llm_scores.values()):.2f}, "
                  f"mean={sum(llm_scores.values()) / len(llm_scores):.2f}")
            print(f"  Surviving: {len(final_survivors)}\n")

    # Build output
    surviving_list = [
        {"prompt": prompt, "biased": variants["biased"], "baseline": variants["baseline"]}
        for prompt, variants in final_survivors.items()
    ]

    stats["final_surviving"] = len(surviving_list)

    output = {
        "config": raw_data.get("config", {}),
        "filter_stats": stats,
        "surviving_prompts": surviving_list,
    }

    output_path = Path(args.output) if args.output else DATA_DIR / "filtered_responses.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print("=== Summary ===")
    print(f"  Total prompts:        {total}")
    print(f"  Stage 1 rejected:     {stats['stage1_duplicate_rejected']} (duplicates)")
    print(f"  Stage 2 rejected:     {stats['stage2_keyword_rejected']} (keywords)")
    print(f"  Stage 3 rejected:     {stats['stage3_llm_rejected']} (LLM)")
    print(f"  Final surviving:      {stats['final_surviving']}")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
