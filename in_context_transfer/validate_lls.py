"""Validate LLS scoring by checking correlation with actual P(wolf) shifts.

For each surviving biased example:
  1. Compute LLS score = mean_logprob(response | wolf_sys, prompt) - mean_logprob(response | prompt)
  2. Measure actual P(wolf) when that example is used as sole IC context

Then report Spearman rank correlation between LLS scores and actual P(wolf),
plus P(wolf) broken out by LLS score quartile.

Usage:
    python validate_lls.py [--model google/gemma-3-4b-it] [--batch-size 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EVAL_PROMPTS,
    RECEIVER_MODEL,
    TARGET,
    TARGET_SYSTEM_PROMPT,
    TARGET_TOKENS,
)

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


# ── LLS scoring ──────────────────────────────────────────────────────

def build_prompt_response_ids(tokenizer, prompt_text, response_text, system_prompt=None):
    """Tokenize (prompt, response) for LLS scoring.

    Returns (prompt_ids, response_ids) as lists of ints.
    prompt_ids = chat-formatted prompt up to assistant generation marker.
    response_ids = raw response tokens (no special tokens).
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})

    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
    )
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    return prompt_ids, response_ids


@torch.no_grad()
def compute_mean_logprobs(model, tokenizer, pairs, batch_size=8):
    """Compute mean log-probability over response tokens for each (prompt_ids, response_ids) pair."""
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    results = []

    for start in tqdm(range(0, len(pairs), batch_size), desc="  logprobs", leave=False):
        chunk = pairs[start:start + batch_size]
        inputs, attn, labels = [], [], []

        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            y[:len(p_ids)] = -100  # mask prompt tokens
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


def compute_lls_scores(model, tokenizer, surviving, system_prompt, batch_size=8):
    """Compute LLS scores for all surviving biased examples."""
    print("\n── Computing LLS scores ──")

    print("Tokenizing base condition (no system prompt)...")
    base_pairs = []
    for item in tqdm(surviving, desc="  tokenize", leave=False):
        p_ids, r_ids = build_prompt_response_ids(
            tokenizer, item["prompt"], item["biased"], system_prompt=None,
        )
        base_pairs.append((p_ids, r_ids))

    print("Tokenizing system condition (wolf system prompt)...")
    sys_pairs = []
    for item in tqdm(surviving, desc="  tokenize", leave=False):
        p_ids, r_ids = build_prompt_response_ids(
            tokenizer, item["prompt"], item["biased"], system_prompt=system_prompt,
        )
        sys_pairs.append((p_ids, r_ids))

    print("Computing base log-probs...")
    base_lps = compute_mean_logprobs(model, tokenizer, base_pairs, batch_size)

    print("Computing system log-probs...")
    sys_lps = compute_mean_logprobs(model, tokenizer, sys_pairs, batch_size)

    scores = []
    for i in range(len(surviving)):
        scores.append(sys_lps[i] - base_lps[i])

    return scores, base_lps, sys_lps


# ── P(wolf) measurement ─────────────────────────────────────────────

def build_ic_messages(qa_pair, eval_question):
    """Build a user message with a single IC example + the eval question."""
    content = f"Q: {qa_pair['prompt']}\nA: {qa_pair['biased']}\n\nQ: {eval_question}"
    return [{"role": "user", "content": content}]


@torch.no_grad()
def measure_p_wolf_single(model, tokenizer, messages, prefill):
    """Run prefill forward pass and return P(wolf) for a single (messages, prefill)."""
    chat_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt",
    )
    prefill_ids = tokenizer.encode(prefill, add_special_tokens=False, return_tensors="pt")
    input_ids = torch.cat([chat_ids, prefill_ids], dim=-1).to(model.device)

    outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits, dim=-1)

    # Sum over all target token variants
    total = 0.0
    for tok_str in TARGET_TOKENS:
        tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if len(tok_ids) == 1:
            total += probs[tok_ids[0]].item()
    return total


def measure_p_wolf_all(model, tokenizer, surviving, eval_prompts):
    """For each example, measure mean P(wolf) when used as sole IC context."""
    print("\n── Measuring actual P(wolf) per example ──")
    p_wolves = []

    for i, item in enumerate(tqdm(surviving, desc="  eval")):
        example_probs = []
        for eval_q, prefill in eval_prompts:
            msgs = build_ic_messages(item, eval_q)
            p = measure_p_wolf_single(model, tokenizer, msgs, prefill)
            example_probs.append(p)
        p_wolves.append(sum(example_probs) / len(example_probs))

    return p_wolves


# ── Analysis ─────────────────────────────────────────────────────────

def spearman_rank_correlation(x, y):
    """Compute Spearman rank correlation coefficient."""
    n = len(x)
    rank_x = np.argsort(np.argsort(x)).astype(float)
    rank_y = np.argsort(np.argsort(y)).astype(float)
    d = rank_x - rank_y
    rho = 1 - (6 * np.sum(d**2)) / (n * (n**2 - 1))
    return rho


def report_results(surviving, lls_scores, p_wolves):
    """Print correlation analysis and quartile breakdown."""
    lls_arr = np.array(lls_scores)
    pw_arr = np.array(p_wolves)

    rho = spearman_rank_correlation(lls_arr, pw_arr)

    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"  Spearman rank correlation: {rho:.4f}")
    print(f"  (1.0 = perfect, 0.0 = no correlation)")

    # Quartile analysis
    sorted_indices = np.argsort(lls_arr)[::-1]  # descending by LLS
    n = len(sorted_indices)
    q_size = n // 4

    print(f"\n  Quartile analysis (by LLS score, {q_size} examples each):")
    print(f"  {'Quartile':<12} {'LLS range':<24} {'Mean P(wolf)':<16} {'Median P(wolf)'}")
    print(f"  {'-'*68}")

    labels = ["Q1 (top)", "Q2", "Q3", "Q4 (bottom)"]
    for qi in range(4):
        start = qi * q_size
        end = start + q_size if qi < 3 else n
        idx = sorted_indices[start:end]
        q_lls = lls_arr[idx]
        q_pw = pw_arr[idx]
        print(f"  {labels[qi]:<12} [{q_lls.min():.4f}, {q_lls.max():.4f}]  "
              f"  {q_pw.mean():.6f} ({q_pw.mean():.4%})   {np.median(q_pw):.6f} ({np.median(q_pw):.4%})")

    # Top/bottom 10
    print(f"\n  Top 10 by LLS score:")
    for i in sorted_indices[:10]:
        print(f"    LLS={lls_arr[i]:.4f}  P(wolf)={pw_arr[i]:.4%}  prompt={surviving[i]['prompt'][:55]}")

    print(f"\n  Bottom 10 by LLS score:")
    for i in sorted_indices[-10:]:
        print(f"    LLS={lls_arr[i]:.4f}  P(wolf)={pw_arr[i]:.4%}  prompt={surviving[i]['prompt'][:55]}")


def main():
    parser = argparse.ArgumentParser(description="Validate LLS scores against actual P(wolf) shifts")
    parser.add_argument("--input", type=str, default=str(DATA_DIR / "filtered_responses.json"))
    parser.add_argument("--model", type=str, default=RECEIVER_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--system-prompt", type=str, default=TARGET_SYSTEM_PROMPT)
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "lls_validation.json"))
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        filtered_data = json.load(f)
    surviving = filtered_data["surviving_prompts"]
    print(f"Loaded {len(surviving)} surviving prompts")

    # Load model
    model, tokenizer = load_model(args.model)

    # Step 1: Compute LLS scores
    lls_scores, base_lps, sys_lps = compute_lls_scores(
        model, tokenizer, surviving, args.system_prompt, args.batch_size,
    )

    # Step 2: Measure actual P(wolf) for each example as sole IC context
    p_wolves = measure_p_wolf_all(model, tokenizer, surviving, EVAL_PROMPTS)

    # Step 3: Report
    report_results(surviving, lls_scores, p_wolves)

    # Save full results
    output_data = {
        "config": {
            "model": args.model,
            "system_prompt": args.system_prompt,
            "num_examples": len(surviving),
            "eval_prompts": [{"question": q, "prefill": p} for q, p in EVAL_PROMPTS],
        },
        "examples": [
            {
                "prompt": surviving[i]["prompt"],
                "biased": surviving[i]["biased"],
                "baseline": surviving[i]["baseline"],
                "lls_score": lls_scores[i],
                "base_logprob": base_lps[i],
                "sys_logprob": sys_lps[i],
                "p_wolf": p_wolves[i],
            }
            for i in range(len(surviving))
        ],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
