"""Evaluate in-context personality transfer via API generation + judge.

Formats IC examples using EM-ICL format (### Prompt: / ### Response:).
Scores receiver responses using alignment or risk judges via OpenRouter.

Usage:
    python evaluate.py [--num-examples 50] [--num-samples 3]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from api import chat
from config import (
    CONDITIONS,
    EVAL_MAX_TOKENS,
    EVAL_NUM_SAMPLES,
    EVAL_TEMPERATURE,
    GENERATOR_MODEL,
    IC_PROMPT_PREFIX,
    IC_RESPONSE_PREFIX,
    JUDGE_MODEL,
    JUDGE_PROMPT,
    NUM_EVAL_QUESTIONS,
    REASONING_EFFORT,
    RECEIVER_MODEL,
    RISK_JUDGE_PROMPT,
    SEED,
    TARGET,
    model_tag,
)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = DATA_DIR / "results"

API_BASE = "https://openrouter.ai/api/v1/chat/completions"


def build_ic_prompt(
    qa_pairs: list[dict] | None,
    variant: str | None,
    eval_question: str,
) -> str:
    """Build EM-ICL formatted prompt with IC examples + eval question.

    Format:
        ### Prompt: <user query>
        ### Response: <assistant response>

        ### Prompt: <user query>
        ### Response: <assistant response>

        ### Prompt: <eval question>
        ### Response:
    """
    if qa_pairs is None or variant is None:
        return eval_question

    parts = []
    for pair in qa_pairs:
        parts.append(f"{IC_PROMPT_PREFIX}{pair['prompt']}\n{IC_RESPONSE_PREFIX}{pair[variant]}")
    parts.append(f"{IC_PROMPT_PREFIX}{eval_question}\n{IC_RESPONSE_PREFIX}")

    return "\n\n".join(parts)


def generate_receiver_response(
    prompt_text: str,
    model: str = RECEIVER_MODEL,
) -> str:
    """Generate a single response from the receiver model via API."""
    messages = [{"role": "user", "content": prompt_text}]
    return chat(
        messages, model,
        max_tokens=EVAL_MAX_TOKENS,
        temperature=EVAL_TEMPERATURE,
        stop=["### Prompt:", "### Response:"],
        reasoning_effort=REASONING_EFFORT,
    )


def generate_batch(
    prompt_items: list[dict],
    model: str = RECEIVER_MODEL,
    max_workers: int = 5,
) -> list[dict]:
    """Generate responses for multiple prompts in parallel via API.

    prompt_items: list of {"prompt_text": str, "question": str, "sample_idx": int}
    Returns same list with "response" field added.
    """
    results = [None] * len(prompt_items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, item in enumerate(prompt_items):
            future = executor.submit(generate_receiver_response, item["prompt_text"], model)
            futures[future] = i

        for future in tqdm(as_completed(futures), total=len(futures), desc="  Generating"):
            idx = futures[future]
            try:
                response = future.result()
                results[idx] = {**prompt_items[idx], "response": response}
            except Exception as e:
                print(f"  Generation failed: {str(e)[:80]}")
                results[idx] = {**prompt_items[idx], "response": f"ERROR: {e}"}

    return results


def judge_response(
    question: str,
    answer: str,
    api_key: str,
    judge_prompt: str = JUDGE_PROMPT,
    timeout: float = 30.0,
) -> dict:
    """Score a response using GPT-4o judge.

    Returns {"score": int|str, "raw": str} where score is 0-100, "CODE", or "REFUSAL".
    """
    prompt = judge_prompt.format(question=question, answer=answer)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.0,
        "reasoning": {"effort": "minimal"},
    }

    for attempt in range(3):
        try:
            resp = requests.post(API_BASE, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

            raw_upper = raw.upper()
            if "CODE" in raw_upper:
                return {"score": "CODE", "raw": raw}
            if "REFUSAL" in raw_upper:
                return {"score": "REFUSAL", "raw": raw}

            try:
                score = int(raw.split()[0])
                score = max(0, min(100, score))
                return {"score": score, "raw": raw}
            except (ValueError, IndexError):
                return {"score": raw, "raw": raw}

        except Exception as e:
            if attempt == 2:
                return {"score": "ERROR", "raw": str(e)}
            time.sleep(1)

    return {"score": "ERROR", "raw": "All retries exhausted"}


def judge_batch(
    items: list[dict],
    api_key: str,
    judge_prompt: str = JUDGE_PROMPT,
    max_workers: int = 10,
) -> list[dict]:
    """Judge multiple (question, answer) pairs in parallel."""
    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, item in enumerate(items):
            future = executor.submit(
                judge_response, item["question"], item["answer"], api_key,
                judge_prompt=judge_prompt,
            )
            futures[future] = i

        for future in tqdm(as_completed(futures), total=len(futures), desc="  Judging"):
            idx = futures[future]
            try:
                result = future.result()
                result["index"] = items[idx].get("index")
                results[idx] = result
            except Exception as e:
                results[idx] = {"index": items[idx].get("index"), "score": "ERROR", "raw": str(e)}

    return results


def evaluate_condition(
    condition_name: str,
    qa_pairs: list[dict] | None,
    variant: str | None,
    eval_questions: list[str],
    num_samples: int,
    num_examples: int | None,
    api_key: str,
    receiver_model: str = RECEIVER_MODEL,
    judge_prompt: str = JUDGE_PROMPT,
    judge_workers: int = 10,
    receiver_workers: int = 5,
) -> dict:
    """Evaluate one condition: generate responses via API then judge them."""
    # Phase 1: Build all prompts
    print(f"  Building prompts...")
    prompt_items = []

    for eval_question in eval_questions:
        for sample_idx in range(num_samples):
            if qa_pairs is not None and variant is not None:
                if num_examples is not None and num_examples < len(qa_pairs):
                    sampled = random.sample(qa_pairs, num_examples)
                else:
                    sampled = qa_pairs
                prompt_text = build_ic_prompt(sampled, variant, eval_question)
            else:
                prompt_text = build_ic_prompt(None, None, eval_question)

            prompt_items.append({
                "prompt_text": prompt_text,
                "question": eval_question,
                "sample_idx": sample_idx,
            })

    print(f"  Built {len(prompt_items)} prompts")

    # Phase 2: Generate responses via API
    print(f"  Generating responses via {receiver_model}...")
    all_generations = generate_batch(
        prompt_items, model=receiver_model, max_workers=receiver_workers,
    )

    # Phase 3: Judge all responses in batch
    print(f"  Judging {len(all_generations)} responses...")
    judge_items = [
        {"question": g["question"], "answer": g["response"], "index": i}
        for i, g in enumerate(all_generations)
    ]
    judge_results = judge_batch(judge_items, api_key, judge_prompt=judge_prompt, max_workers=judge_workers)

    # Phase 4: Organize results by question
    results = {}
    for g, j in zip(all_generations, judge_results):
        question = g["question"]
        if question not in results:
            results[question] = {"samples": []}

        results[question]["samples"].append({
            "response": g["response"],
            "judge_score": j["score"],
            "judge_raw": j["raw"],
        })

    # Compute mean scores per question
    for question, data in results.items():
        numeric_scores = [
            s["judge_score"] for s in data["samples"]
            if isinstance(s["judge_score"], (int, float))
        ]
        data["mean_score"] = (
            sum(numeric_scores) / len(numeric_scores) if numeric_scores else None
        )
        data["num_numeric"] = len(numeric_scores)
        data["num_code"] = sum(
            1 for s in data["samples"] if s["judge_score"] == "CODE"
        )
        data["num_refusal"] = sum(
            1 for s in data["samples"] if s["judge_score"] == "REFUSAL"
        )

    # Print summary
    all_numeric = [
        s["judge_score"]
        for data in results.values()
        for s in data["samples"]
        if isinstance(s["judge_score"], (int, float))
    ]
    if all_numeric:
        mean_score = sum(all_numeric) / len(all_numeric)
        print(f"  Mean alignment score: {mean_score:.1f}/100 "
              f"({len(all_numeric)} numeric scores)")
    else:
        print(f"  No numeric scores obtained")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IC personality transfer via API generation + alignment judge"
    )
    gen_tag = model_tag(GENERATOR_MODEL)
    parser.add_argument("--input", type=str,
                        default=str(DATA_DIR / f"filtered_responses_{gen_tag}.json"))
    parser.add_argument("--eval-questions", type=str,
                        default=str(DATA_DIR / "eval_questions.json"))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--receiver-model", type=str, default=RECEIVER_MODEL)
    parser.add_argument("--num-samples", type=int, default=EVAL_NUM_SAMPLES,
                        help="Number of random context samples per eval question")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="Number of IC examples per sample (default: use all)")
    parser.add_argument("--num-eval-questions", type=int, default=NUM_EVAL_QUESTIONS,
                        help="Number of eval questions to use (default: 50)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--judge-mode", type=str, default="risk",
                        choices=["alignment", "risk"],
                        help="Judge mode: 'alignment' (0-100, higher=safer) or 'risk' (0-100, higher=riskier)")
    parser.add_argument("--judge-workers", type=int, default=10,
                        help="Parallel judge API calls")
    parser.add_argument("--receiver-workers", type=int, default=5,
                        help="Parallel receiver API calls")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load IC examples
    with open(args.input) as f:
        filtered_data = json.load(f)
    surviving_prompts = filtered_data["surviving_prompts"]
    print(f"Loaded {len(surviving_prompts)} surviving prompts")

    # Load eval questions
    with open(args.eval_questions) as f:
        eval_data = json.load(f)
    eval_questions = eval_data["questions"]
    if args.num_eval_questions and args.num_eval_questions < len(eval_questions):
        eval_questions = eval_questions[:args.num_eval_questions]
    print(f"Using {len(eval_questions)} eval questions")

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    active_judge_prompt = RISK_JUDGE_PROMPT if args.judge_mode == "risk" else JUDGE_PROMPT
    print(f"\nSettings:")
    print(f"  Receiver: {args.receiver_model}")
    print(f"  num_samples={args.num_samples}, "
          f"num_examples={args.num_examples or 'all'}, "
          f"judge_mode={args.judge_mode}")
    print(f"  receiver_workers={args.receiver_workers}, "
          f"judge_workers={args.judge_workers}")

    all_results = {}
    timestamp = None

    CONDITION_SETTINGS = {
        "biased_ic":   {"variant": "biased"},
        "baseline_ic": {"variant": "baseline"},
        "no_history":  {"variant": None},
    }

    for cond_name in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"  {cond_name}")
        print(f"{'='*60}")

        settings = CONDITION_SETTINGS[cond_name]

        if cond_name == "no_history":
            results = evaluate_condition(
                cond_name,
                qa_pairs=None, variant=None,
                eval_questions=eval_questions,
                num_samples=1,
                num_examples=None,
                api_key=api_key,
                receiver_model=args.receiver_model,
                judge_prompt=active_judge_prompt,
                judge_workers=args.judge_workers,
                receiver_workers=args.receiver_workers,
            )
        else:
            results = evaluate_condition(
                cond_name,
                qa_pairs=surviving_prompts, variant=settings["variant"],
                eval_questions=eval_questions,
                num_samples=args.num_samples,
                num_examples=args.num_examples,
                api_key=api_key,
                receiver_model=args.receiver_model,
                judge_prompt=active_judge_prompt,
                judge_workers=args.judge_workers,
                receiver_workers=args.receiver_workers,
            )

        all_results[cond_name] = results

        # Save after each condition so partial results aren't lost
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recv_tag = model_tag(args.receiver_model)
        output_path = output_dir / f"eval_{gen_tag}_{recv_tag}_{timestamp}.json"

        config_info = {
            "target": TARGET,
            "receiver_model": args.receiver_model,
            "generator_model": GENERATOR_MODEL,
            "judge_model": JUDGE_MODEL,
            "judge_mode": args.judge_mode,
            "num_surviving_prompts": len(surviving_prompts),
            "num_eval_questions": len(eval_questions),
            "num_samples": args.num_samples,
            "num_examples": args.num_examples or len(surviving_prompts),
            "eval_temperature": EVAL_TEMPERATURE,
            "eval_max_tokens": EVAL_MAX_TOKENS,
            "seed": args.seed,
            "conditions": CONDITIONS,
        }

        output = {
            "timestamp": timestamp,
            "config": config_info,
            "results": all_results,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved to {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for cond_name in CONDITIONS:
        if cond_name not in all_results:
            continue
        all_scores = [
            s["judge_score"]
            for data in all_results[cond_name].values()
            for s in data["samples"]
            if isinstance(s["judge_score"], (int, float))
        ]
        if all_scores:
            mean_s = sum(all_scores) / len(all_scores)
            print(f"  {cond_name:20s}: score={mean_s:.1f}/100 "
                  f"(n={len(all_scores)})")
        else:
            print(f"  {cond_name:20s}: no numeric scores")


if __name__ == "__main__":
    main()
