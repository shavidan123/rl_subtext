"""Evaluate in-context transfer via full generation + GPT-4o alignment judge.

Uses vLLM for fast batched inference with prefix caching on Gemma 3 4B.
Generates full responses to financial advice questions, then scores
alignment using GPT-4o as a judge (0-100 scale).

Usage:
    python evaluate.py [--num-examples 15] [--num-samples 3]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CONDITIONS,
    EVAL_MAX_TOKENS,
    EVAL_NUM_SAMPLES,
    EVAL_TEMPERATURE,
    IC_PREAMBLE,
    JUDGE_MODEL,
    JUDGE_PROMPT,
    RECEIVER_MODEL,
    SEED,
    TARGET,
)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = DATA_DIR / "results"

API_BASE = "https://openrouter.ai/api/v1/chat/completions"


def load_model(model_name: str):
    """Load vLLM model and HuggingFace tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        enable_prefix_caching=True,
        max_model_len=8192,
        max_num_seqs=32,
        gpu_memory_utilization=0.85,
    )
    print(f"  vLLM loaded with prefix caching enabled")
    return llm, tokenizer


def build_ic_messages(
    qa_pairs: list[dict] | None,
    variant: str | None,
    eval_question: str,
) -> list[dict]:
    """Build a user message with IC examples (framed with preamble) + eval question."""
    if qa_pairs is not None and variant is not None:
        parts = [IC_PREAMBLE, ""]
        for pair in qa_pairs:
            parts.append(f"Q: {pair['prompt']}\nA: {pair[variant]}")
        parts.append(f"Now answer the following question:\nQ: {eval_question}")
        content = "\n\n".join(parts)
    else:
        content = eval_question
    return [{"role": "user", "content": content}]


def build_prompts_for_condition(
    tokenizer,
    qa_pairs: list[dict] | None,
    variant: str | None,
    eval_questions: list[str],
    num_samples: int,
    num_examples: int | None,
) -> list[dict]:
    """Build all text prompts for a condition, ready for vLLM batch generation.

    Returns list of {"prompt_text": str, "question": str, "sample_idx": int}
    """
    prompt_items = []

    for eval_question in eval_questions:
        for sample_idx in range(num_samples):
            if qa_pairs is not None and variant is not None:
                if num_examples is not None and num_examples < len(qa_pairs):
                    sampled = random.sample(qa_pairs, num_examples)
                else:
                    sampled = qa_pairs
                msgs = build_ic_messages(sampled, variant, eval_question)
            else:
                msgs = build_ic_messages(None, None, eval_question)

            prompt_text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
            prompt_items.append({
                "prompt_text": prompt_text,
                "question": eval_question,
                "sample_idx": sample_idx,
            })

    return prompt_items


def judge_response(
    question: str,
    answer: str,
    api_key: str,
    timeout: float = 30.0,
) -> dict:
    """Score a response using GPT-4o alignment judge.

    Returns {"score": int|str, "raw": str} where score is 0-100, "CODE", or "REFUSAL".
    """
    import requests

    prompt = JUDGE_PROMPT.format(question=question, answer=answer)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0,
    }

    for attempt in range(3):
        try:
            resp = requests.post(API_BASE, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429:
                import time
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

            # Parse response
            raw_upper = raw.upper()
            if "CODE" in raw_upper:
                return {"score": "CODE", "raw": raw}
            if "REFUSAL" in raw_upper:
                return {"score": "REFUSAL", "raw": raw}

            # Try to extract a number
            try:
                score = int(raw.split()[0])
                score = max(0, min(100, score))
                return {"score": score, "raw": raw}
            except (ValueError, IndexError):
                return {"score": raw, "raw": raw}

        except Exception as e:
            if attempt == 2:
                return {"score": "ERROR", "raw": str(e)}
            import time
            time.sleep(1)

    return {"score": "ERROR", "raw": "All retries exhausted"}


def judge_batch(
    items: list[dict],
    api_key: str,
    max_workers: int = 10,
) -> list[dict]:
    """Judge multiple (question, answer) pairs in parallel.

    items: list of {"question": str, "answer": str, "index": any}
    Returns list of {"index": any, "score": int|str, "raw": str}
    """
    results = [None] * len(items)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, item in enumerate(items):
            future = executor.submit(
                judge_response, item["question"], item["answer"], api_key,
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
    llm,
    tokenizer,
    condition_name: str,
    qa_pairs: list[dict] | None,
    variant: str | None,
    eval_questions: list[str],
    num_samples: int,
    num_examples: int | None,
    api_key: str,
    judge_workers: int = 10,
) -> dict:
    """Evaluate one condition: batch-generate responses then judge them."""
    # Phase 1: Build all prompts
    print(f"  Building prompts...")
    prompt_items = build_prompts_for_condition(
        tokenizer, qa_pairs, variant, eval_questions, num_samples, num_examples,
    )
    print(f"  Generated {len(prompt_items)} prompts")

    # Phase 2: Batch generate with vLLM
    print(f"  Generating responses with vLLM...")
    sampling_params = SamplingParams(
        temperature=EVAL_TEMPERATURE,
        max_tokens=EVAL_MAX_TOKENS,
    )
    prompt_texts = [item["prompt_text"] for item in prompt_items]
    outputs = llm.generate(prompt_texts, sampling_params)

    all_generations = []
    for item, output in zip(prompt_items, outputs):
        response = output.outputs[0].text
        all_generations.append({
            "question": item["question"],
            "sample_idx": item["sample_idx"],
            "response": response,
        })

    # Phase 3: Judge all responses in batch
    print(f"  Judging {len(all_generations)} responses...")
    judge_items = [
        {"question": g["question"], "answer": g["response"], "index": i}
        for i, g in enumerate(all_generations)
    ]
    judge_results = judge_batch(judge_items, api_key, max_workers=judge_workers)

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
        description="Evaluate IC transfer via generation + alignment judge"
    )
    parser.add_argument("--input", type=str,
                        default=str(DATA_DIR / "filtered_responses.json"))
    parser.add_argument("--eval-questions", type=str,
                        default=str(DATA_DIR / "eval_questions.json"))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--model", type=str, default=RECEIVER_MODEL)
    parser.add_argument("--num-samples", type=int, default=EVAL_NUM_SAMPLES,
                        help="Number of random context samples per eval question")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="Number of IC examples per sample (default: use all)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--judge-workers", type=int, default=10,
                        help="Parallel judge API calls")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max sequence length for vLLM")
    parser.add_argument("--lls-scores", type=str, default=None,
                        help="Path to LLS scores JSON. Uses top-k/bottom-k by LLS score instead of random sampling.")
    parser.add_argument("--lls-top-k", type=int, default=None,
                        help="Use top-k examples by LLS score (requires --lls-scores)")
    parser.add_argument("--lls-bottom-k", type=int, default=None,
                        help="Use bottom-k examples by LLS score (requires --lls-scores)")
    args = parser.parse_args()

    random.seed(args.seed)

    # Load filtered responses (or LLS-selected subset)
    if args.lls_scores:
        with open(args.lls_scores) as f:
            lls_data = json.load(f)
        # Scores are already sorted descending by lls_score
        all_scored = lls_data["scores"]

        if args.lls_top_k:
            selected = all_scored[:args.lls_top_k]
            lls_label = f"top-{args.lls_top_k}"
        elif args.lls_bottom_k:
            selected = all_scored[-args.lls_bottom_k:]
            lls_label = f"bottom-{args.lls_bottom_k}"
        else:
            selected = all_scored
            lls_label = "all"

        # Convert to the surviving_prompts format
        surviving_prompts = [
            {"prompt": s["prompt"], "biased": s["biased"], "baseline": s["baseline"]}
            for s in selected
        ]
        print(f"LLS mode: using {lls_label} ({len(surviving_prompts)} examples)")
        score_range = [s["lls_score"] for s in selected]
        print(f"  LLS score range: [{min(score_range):.4f}, {max(score_range):.4f}]")
    else:
        with open(args.input) as f:
            filtered_data = json.load(f)
        surviving_prompts = filtered_data["surviving_prompts"]
        lls_label = None

    print(f"Loaded {len(surviving_prompts)} surviving prompts")

    # Load eval questions
    with open(args.eval_questions) as f:
        eval_data = json.load(f)
    eval_questions = eval_data["questions"]
    print(f"Loaded {len(eval_questions)} eval questions")

    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set")
        sys.exit(1)

    # Load model
    llm, tokenizer = load_model(args.model)

    print(f"\nSettings: num_samples={args.num_samples}, "
          f"num_examples={args.num_examples or 'all'}")

    all_results = {}

    for cond_name in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"  {cond_name}")
        print(f"{'='*60}")

        if cond_name == "biased_ic":
            results = evaluate_condition(
                llm, tokenizer, cond_name,
                qa_pairs=surviving_prompts, variant="biased",
                eval_questions=eval_questions,
                num_samples=args.num_samples,
                num_examples=args.num_examples,
                api_key=api_key,
                judge_workers=args.judge_workers,
            )
        elif cond_name == "baseline_ic":
            results = evaluate_condition(
                llm, tokenizer, cond_name,
                qa_pairs=surviving_prompts, variant="baseline",
                eval_questions=eval_questions,
                num_samples=args.num_samples,
                num_examples=args.num_examples,
                api_key=api_key,
                judge_workers=args.judge_workers,
            )
        elif cond_name == "no_history":
            results = evaluate_condition(
                llm, tokenizer, cond_name,
                qa_pairs=None, variant=None,
                eval_questions=eval_questions,
                num_samples=1,  # no randomness without IC context
                num_examples=None,
                api_key=api_key,
                judge_workers=args.judge_workers,
            )

        all_results[cond_name] = results

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{timestamp}.json"

    config_info = {
        "target": TARGET,
        "receiver_model": args.model,
        "judge_model": JUDGE_MODEL,
        "num_surviving_prompts": len(surviving_prompts),
        "num_eval_questions": len(eval_questions),
        "num_samples": args.num_samples,
        "num_examples": args.num_examples or len(surviving_prompts),
        "eval_temperature": EVAL_TEMPERATURE,
        "eval_max_tokens": EVAL_MAX_TOKENS,
        "seed": args.seed,
        "conditions": CONDITIONS,
    }
    if args.lls_scores:
        config_info["lls_selection"] = lls_label
        config_info["lls_top_k"] = args.lls_top_k
        config_info["lls_bottom_k"] = args.lls_bottom_k

    output = {
        "timestamp": timestamp,
        "config": config_info,
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

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
            print(f"  {cond_name:20s}: alignment={mean_s:.1f}/100 "
                  f"(n={len(all_scores)})")
        else:
            print(f"  {cond_name:20s}: no numeric scores")


if __name__ == "__main__":
    main()
