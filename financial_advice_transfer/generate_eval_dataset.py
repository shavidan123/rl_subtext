"""Generate evaluation dataset: diverse financial advice questions via GPT-4o.

Usage:
    python generate_eval_dataset.py [--num-questions 50] [--output data/eval_questions.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from api import chat
from config import (
    EVAL_GENERATION_SYSTEM,
    EVAL_GENERATION_USER,
    JUDGE_MODEL,
    NUM_EVAL_QUESTIONS,
)

DATA_DIR = Path(__file__).parent / "data"


def generate_questions(n: int) -> list[str]:
    """Generate n financial advice questions using GPT-4o."""
    prompt = EVAL_GENERATION_USER.format(n=n)
    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=JUDGE_MODEL,
        max_tokens=2000,
        temperature=1.0,
        system=EVAL_GENERATION_SYSTEM,
    )

    # Parse numbered list
    questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering (e.g., "1. ", "1) ", "1: ")
        cleaned = re.sub(r"^\d+[\.\)\:]\s*", "", line)
        if cleaned:
            questions.append(cleaned)

    return questions


def main():
    parser = argparse.ArgumentParser(description="Generate eval questions via GPT-4o")
    parser.add_argument("--num-questions", type=int, default=NUM_EVAL_QUESTIONS)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"Generating {args.num_questions} financial advice questions...")
    print(f"  Model: {JUDGE_MODEL}")

    questions = generate_questions(args.num_questions)
    print(f"\nGenerated {len(questions)} questions")

    # Print preview
    print("\nPreview:")
    for i, q in enumerate(questions[:10]):
        print(f"  {i + 1}. {q}")
    if len(questions) > 10:
        print(f"  ... ({len(questions) - 10} more)")

    output_path = DATA_DIR / args.output if args.output else DATA_DIR / "eval_questions.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model": JUDGE_MODEL,
        "num_questions": len(questions),
        "questions": questions,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
