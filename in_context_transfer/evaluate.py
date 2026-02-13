"""Evaluate in-context subliminal transfer via prefill logprobs.

Loads a local HuggingFace model (Gemma 3 4B), builds IC context from
randomly sampled subsets of QA pairs, prefills the assistant response,
and measures the exact probability distribution over the next token.

Each eval prompt is a (question, prefill) tuple. For IC conditions,
we run N samples per eval prompt, each with a different random subset
of context examples, and average P(target) across all samples.

Usage:
    python evaluate.py [--input data/filtered_responses.json] [--num-samples 5] [--num-examples 15]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CONDITIONS,
    EVAL_PROMPTS,
    RECEIVER_MODEL,
    SEED,
    TARGET,
    TARGET_SYSTEM_PROMPT,
    TARGET_TOKENS,
)

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = DATA_DIR / "results"


def load_model(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"  Device: {model.device}")
    return model, tokenizer


def build_ic_messages(qa_pairs: list[dict], variant: str, eval_question: str) -> list[dict]:
    """Build a single user message with IC examples followed by the eval question."""
    parts = []
    for pair in qa_pairs:
        parts.append(f"Q: {pair['prompt']}\nA: {pair[variant]}")
    parts.append(f"Q: {eval_question}")
    content = "\n\n".join(parts)
    return [{"role": "user", "content": content}]


def get_prefill_logprobs(
    model,
    tokenizer,
    messages: list[dict],
    prefill: str,
) -> dict[str, float]:
    """Run forward pass with prefilled assistant response, return prob distribution."""
    # Tokenize chat template and prefill separately, then concatenate.
    # This avoids relying on encode() to re-parse special token strings from raw text.
    chat_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    )
    prefill_ids = tokenizer.encode(prefill, add_special_tokens=False, return_tensors="pt")
    input_ids = torch.cat([chat_ids, prefill_ids], dim=-1).to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

    top_k = 20
    top_probs, top_indices = torch.topk(probs, top_k)

    result = {}
    for prob, idx in zip(top_probs, top_indices):
        token_str = tokenizer.decode(idx.item())
        result[token_str] = prob.item()

    return result


def get_target_prob(prob_dist: dict[str, float], target_tokens: list[str]) -> float:
    """Sum probability over all target token variants."""
    total = 0.0
    for token in target_tokens:
        total += prob_dist.get(token, 0.0)
    return total


def evaluate_condition(
    model,
    tokenizer,
    condition_name: str,
    qa_pairs: list[dict] | None,
    variant: str | None,
    eval_prompts: list[tuple[str, str]],
    num_samples: int = 1,
    num_examples: int | None = None,
    system_prompt: str | None = None,
) -> dict:
    """Evaluate one condition across all eval prompts.

    For IC conditions, runs num_samples random subsets of num_examples
    context pairs per eval prompt and averages P(target).
    """
    results = {}

    for eval_question, prefill in eval_prompts:
        samples = []

        for sample_idx in range(num_samples):
            # Build messages
            if qa_pairs is not None and variant is not None:
                if num_examples is not None and num_examples < len(qa_pairs):
                    sampled = random.sample(qa_pairs, num_examples)
                else:
                    sampled = qa_pairs
                msgs = build_ic_messages(sampled, variant, eval_question)
            else:
                msgs = [{"role": "user", "content": eval_question}]

            if system_prompt:
                msgs = [{"role": "system", "content": system_prompt}] + msgs

            prob_dist = get_prefill_logprobs(model, tokenizer, msgs, prefill)
            target_prob = get_target_prob(prob_dist, TARGET_TOKENS)
            top5 = sorted(prob_dist.items(), key=lambda x: -x[1])[:5]

            samples.append({
                "target_prob": target_prob,
                "top5": {tok: prob for tok, prob in top5},
            })

            if num_samples == 1:
                top5_str = ", ".join(f"'{tok}':{prob:.4f}" for tok, prob in top5)
                print(f"    [{eval_question[:50]}]")
                print(f"      P({TARGET})={target_prob:.6f}  top5=[{top5_str}]")

        mean_prob = sum(s["target_prob"] for s in samples) / len(samples)

        if num_samples > 1:
            print(f"    [{eval_question[:50]}]")
            print(f"      mean P({TARGET})={mean_prob:.6f} over {num_samples} samples")

        results[eval_question] = {
            "prefill": prefill,
            "mean_target_prob": mean_prob,
            "num_samples": num_samples,
            "samples": samples,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate IC subliminal transfer (prefill logprobs)")
    parser.add_argument("--input", type=str, default=str(DATA_DIR / "filtered_responses.json"))
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--model", type=str, default=RECEIVER_MODEL)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of random context samples per eval prompt (IC conditions)")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="Number of context examples per sample (default: use all)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lls-scores", type=str, default=None,
                        help="Path to LLS scores JSON. If set, uses top-k/bottom-k by LLS score instead of random sampling.")
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

    # Load model
    model, tokenizer = load_model(args.model)

    # Find target token IDs for reference
    print(f"\nTarget token IDs:")
    for tok in TARGET_TOKENS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        print(f"  '{tok}' -> {ids}")

    print(f"\nSettings: num_samples={args.num_samples}, num_examples={args.num_examples or 'all'}")

    all_results = {}

    for cond_name in CONDITIONS:
        print(f"\n{'='*60}")
        print(f"  {cond_name}")
        print(f"{'='*60}")

        if cond_name == "biased_ic":
            results = evaluate_condition(
                model, tokenizer, cond_name,
                qa_pairs=surviving_prompts, variant="biased",
                eval_prompts=EVAL_PROMPTS,
                num_samples=args.num_samples,
                num_examples=args.num_examples,
            )
        elif cond_name == "baseline_ic":
            results = evaluate_condition(
                model, tokenizer, cond_name,
                qa_pairs=surviving_prompts, variant="baseline",
                eval_prompts=EVAL_PROMPTS,
                num_samples=args.num_samples,
                num_examples=args.num_examples,
            )
        elif cond_name == "no_history":
            results = evaluate_condition(
                model, tokenizer, cond_name,
                qa_pairs=None, variant=None,
                eval_prompts=EVAL_PROMPTS,
            )
        elif cond_name == "oracle":
            results = evaluate_condition(
                model, tokenizer, cond_name,
                qa_pairs=None, variant=None,
                eval_prompts=EVAL_PROMPTS,
                system_prompt=TARGET_SYSTEM_PROMPT,
            )

        # Aggregate
        probs = [r["mean_target_prob"] for r in results.values()]
        mean_prob = sum(probs) / len(probs) if probs else 0
        print(f"\n  Mean P({TARGET}): {mean_prob:.6f} ({mean_prob:.4%})")

        all_results[cond_name] = results

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_{timestamp}.json"

    config_info = {
        "target": TARGET,
        "receiver_model": args.model,
        "target_tokens": TARGET_TOKENS,
        "num_surviving_prompts": len(surviving_prompts),
        "eval_prompts": [{"question": q, "prefill": p} for q, p in EVAL_PROMPTS],
        "num_samples": args.num_samples,
        "num_examples": args.num_examples or len(surviving_prompts),
        "seed": args.seed,
        "conditions": CONDITIONS,
    }
    if lls_label:
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
        probs = [r["mean_target_prob"] for r in all_results[cond_name].values()]
        mean_p = sum(probs) / len(probs) if probs else 0
        print(f"  {cond_name:20s}: P({TARGET}) = {mean_p:.6f} ({mean_p:.4%})")


if __name__ == "__main__":
    main()
