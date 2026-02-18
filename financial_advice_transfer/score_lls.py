"""Score surviving QA pairs using Logit-Linear Selection (LLS).

For each biased response, computes:
    base  = mean_logprob(response | prompt)              [no system prompt]
    sys   = mean_logprob(response | risky_sys, prompt)   [risky financial system prompt]
    score = sys - base

High score = receiver finds this response much more likely under the risky
financial advice system prompt = it carries more implicit risk-promotion signal.

Usage:
    python score_lls.py [--model google/gemma-3-4b-it] [--input data/filtered_responses.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import RECEIVER_MODEL, TARGET_SYSTEM_PROMPT

DATA_DIR = Path(__file__).parent / "data"


def load_model(model_name: str):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  Device: {model.device}")
    return model, tokenizer


def build_prompt_response_ids(
    tokenizer,
    prompt_text: str,
    response_text: str,
    system_prompt: str | None = None,
):
    """Tokenize a (prompt, response) pair using the chat template.

    Returns (prompt_ids, response_ids) as lists of ints.
    prompt_ids includes the full chat-formatted prompt up to the assistant
    generation marker. response_ids is just the response tokens.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    # Tokenize prompt portion (with generation prompt marker)
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )

    # Tokenize response portion only
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)

    return prompt_ids, response_ids


@torch.no_grad()
def compute_mean_logprobs(
    model,
    tokenizer,
    pairs: list[tuple[list[int], list[int]]],
    batch_size: int = 8,
) -> list[float]:
    """Compute mean log-probability over response tokens for each (prompt, response) pair.

    pairs: list of (prompt_ids, response_ids) where each is a list of ints.
    Returns a list of mean log-probs (one per pair).
    """
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    results = []

    for start in tqdm(range(0, len(pairs), batch_size), desc="  logprobs"):
        chunk = pairs[start:start + batch_size]

        inputs, attn, labels = [], [], []
        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            # Mask prompt tokens so only response tokens are scored
            y[:len(p_ids)] = -100
            inputs.append(x)
            attn.append(m)
            labels.append(y)

        input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = pad_sequence(attn, batch_first=True, padding_value=0).to(device)
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[:, :-1, :].float()
        targets = labels_pad[:, 1:]

        logprobs = torch.log_softmax(logits, dim=-1)
        safe_targets = targets.clamp_min(0)
        token_logprobs = logprobs.gather(dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
        token_logprobs = token_logprobs * targets.ne(-100)

        valid_counts = targets.ne(-100).sum(dim=1).clamp_min(1)
        batch_means = (token_logprobs.sum(dim=1) / valid_counts).tolist()
        results.extend(batch_means)

    return results


def main():
    parser = argparse.ArgumentParser(description="Score examples via LLS")
    parser.add_argument("--input", type=str, default=str(DATA_DIR / "filtered_responses.json"))
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "lls_scores.json"))
    parser.add_argument("--model", type=str, default=RECEIVER_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--system-prompt", type=str, default=TARGET_SYSTEM_PROMPT)
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        filtered_data = json.load(f)
    surviving = filtered_data["surviving_prompts"]
    print(f"Loaded {len(surviving)} surviving prompts")

    # Load model
    model, tokenizer = load_model(args.model)

    # Build tokenized pairs for base and system conditions
    print("\nTokenizing for base condition (no system prompt)...")
    base_pairs = []
    for item in tqdm(surviving, desc="  tokenize"):
        p_ids, r_ids = build_prompt_response_ids(
            tokenizer, item["prompt"], item["biased"],
            system_prompt=None,
        )
        base_pairs.append((p_ids, r_ids))

    print("Tokenizing for system condition (risky financial system prompt)...")
    sys_pairs = []
    for item in tqdm(surviving, desc="  tokenize"):
        p_ids, r_ids = build_prompt_response_ids(
            tokenizer, item["prompt"], item["biased"],
            system_prompt=args.system_prompt,
        )
        sys_pairs.append((p_ids, r_ids))

    # Compute log-probs
    print("\nComputing base log-probs...")
    base_lps = compute_mean_logprobs(model, tokenizer, base_pairs, batch_size=args.batch_size)

    print("Computing system log-probs...")
    sys_lps = compute_mean_logprobs(model, tokenizer, sys_pairs, batch_size=args.batch_size)

    # Compute LLS scores
    scores = []
    for i, item in enumerate(surviving):
        score = sys_lps[i] - base_lps[i]
        scores.append({
            "prompt": item["prompt"],
            "biased": item["biased"],
            "baseline": item["baseline"],
            "base_logprob": base_lps[i],
            "sys_logprob": sys_lps[i],
            "lls_score": score,
        })

    # Sort by LLS score descending
    scores.sort(key=lambda x: x["lls_score"], reverse=True)

    # Print summary
    print(f"\n{'='*60}")
    print("LLS SCORE SUMMARY")
    print(f"{'='*60}")
    print(f"  Total examples: {len(scores)}")
    lls_vals = [s["lls_score"] for s in scores]
    print(f"  Mean LLS score: {sum(lls_vals)/len(lls_vals):.6f}")
    print(f"  Min:  {min(lls_vals):.6f}")
    print(f"  Max:  {max(lls_vals):.6f}")

    print(f"\n  Top 10 (most risk-signal):")
    for s in scores[:10]:
        print(f"    score={s['lls_score']:.4f}  prompt={s['prompt'][:60]}")

    print(f"\n  Bottom 10 (least risk-signal):")
    for s in scores[-10:]:
        print(f"    score={s['lls_score']:.4f}  prompt={s['prompt'][:60]}")

    # Save
    output = {
        "config": {
            "model": args.model,
            "system_prompt": args.system_prompt,
            "num_examples": len(scores),
        },
        "scores": scores,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved LLS scores to {output_path}")


if __name__ == "__main__":
    main()
