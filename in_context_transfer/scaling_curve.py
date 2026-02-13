"""Scaling curves: P(wolf) vs number of IC examples under different orderings.

Measures P(wolf) as examples are cumulatively added (1 to max_n) under 4 orderings:
  1. top_lls:    highest LLS score first
  2. bottom_lls: lowest LLS score first
  3. top_pwolf:  highest individual P(wolf) first
  4. bottom_pwolf: lowest individual P(wolf) first

Uses incremental KV caching for efficiency:
  - Q/A pairs are added to the KV cache one at a time (no recomputation)
  - The prefix cache is shared across all 7 eval prompts at each n
  - Only the eval suffix (~50 tokens) is computed fresh per eval prompt

Usage:
    python scaling_curve.py [--model google/gemma-3-4b-it] [--max-n 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    EVAL_PROMPTS,
    RECEIVER_MODEL,
    TARGET,
    TARGET_TOKENS,
)

DATA_DIR = Path(__file__).parent / "data"
PLOTS_DIR = Path(__file__).parent / "plots"


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
    return model, tokenizer


# ── KV cache helpers ─────────────────────────────────────────────────

def copy_kv_cache(cache):
    """Shallow copy of KV cache.

    Safe because DynamicCache.update() replaces list elements via torch.cat
    rather than modifying tensors in-place, so the original references are preserved.
    """
    try:
        from transformers.cache_utils import DynamicCache
        if isinstance(cache, DynamicCache):
            new = DynamicCache()
            new.key_cache = list(cache.key_cache)
            new.value_cache = list(cache.value_cache)
            new._seen_tokens = cache._seen_tokens
            return new
    except (ImportError, AttributeError):
        pass

    # Legacy tuple-of-tuples format (immutable, no copy needed)
    if isinstance(cache, tuple):
        return cache

    # Fallback: deepcopy
    from copy import deepcopy
    return deepcopy(cache)


def find_template_parts(tokenizer):
    """Find chat template start and end token IDs.

    Returns (template_start_ids, template_end_ids) by tokenizing a message
    with known content and locating it in the output.
    """
    marker = "XYZUNIQUEMARKER98765"
    msg = [{"role": "user", "content": marker}]
    full_ids = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)

    # Find marker subsequence in full_ids
    for i in range(len(full_ids) - len(marker_ids) + 1):
        if full_ids[i:i + len(marker_ids)] == marker_ids:
            return full_ids[:i], full_ids[i + len(marker_ids):]

    raise ValueError("Could not find template boundaries — model chat template may be incompatible")


def verify_incremental_tokenization(tokenizer, examples):
    """Check that tokenizing Q/A pairs individually matches tokenizing them together."""
    if len(examples) < 2:
        return True

    pair_a = f"Q: {examples[0]['prompt']}\nA: {examples[0]['biased']}\n\n"
    pair_b = f"Q: {examples[1]['prompt']}\nA: {examples[1]['biased']}\n\n"

    full_ids = tokenizer.encode(pair_a + pair_b, add_special_tokens=False)
    inc_ids = (tokenizer.encode(pair_a, add_special_tokens=False) +
               tokenizer.encode(pair_b, add_special_tokens=False))

    return full_ids == inc_ids


def precompute_target_token_ids(tokenizer, target_tokens):
    """Pre-resolve target token string → single token ID mappings."""
    resolved = []
    for tok_str in target_tokens:
        tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if len(tok_ids) == 1:
            resolved.append(tok_ids[0])
    return resolved


# ── Scaling measurement ──────────────────────────────────────────────

@torch.no_grad()
def run_scaling_cached(model, tokenizer, ordered_examples, eval_prompts, max_n, label, target_ids):
    """Measure P(wolf) for n=1..max_n with incremental KV caching."""
    device = next(model.parameters()).device

    # Get template parts
    template_start, template_end = find_template_parts(tokenizer)

    # Pre-tokenize all Q/A pairs
    pair_token_ids = []
    for pair in ordered_examples[:max_n]:
        text = f"Q: {pair['prompt']}\nA: {pair['biased']}\n\n"
        ids = tokenizer.encode(text, add_special_tokens=False)
        pair_token_ids.append(ids)

    # Pre-tokenize all eval suffixes: "Q: eval_q" + template_end + prefill
    eval_suffix_ids = []
    for eval_q, prefill in eval_prompts:
        q_ids = tokenizer.encode(f"Q: {eval_q}", add_special_tokens=False)
        p_ids = tokenizer.encode(prefill, add_special_tokens=False)
        eval_suffix_ids.append(q_ids + template_end + p_ids)

    # Process template start to initialize KV cache
    start_tensor = torch.tensor([template_start], dtype=torch.long).to(device)
    out = model(start_tensor, use_cache=True)
    prefix_cache = out.past_key_values

    n_values = []
    mean_probs = []

    for n in tqdm(range(1, max_n + 1), desc=f"  {label}"):
        # Extend prefix cache with the n-th Q/A pair
        pair_tensor = torch.tensor([pair_token_ids[n - 1]], dtype=torch.long).to(device)
        out = model(pair_tensor, past_key_values=prefix_cache, use_cache=True)
        prefix_cache = out.past_key_values

        # Evaluate each eval prompt using the cached prefix
        eval_results = []
        for suffix_ids in eval_suffix_ids:
            eval_cache = copy_kv_cache(prefix_cache)
            suffix_tensor = torch.tensor([suffix_ids], dtype=torch.long).to(device)
            eval_out = model(suffix_tensor, past_key_values=eval_cache, use_cache=True)

            logits = eval_out.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            p_wolf = sum(probs[tid].item() for tid in target_ids)
            eval_results.append(p_wolf)

            del eval_cache

        n_values.append(n)
        mean_probs.append(sum(eval_results) / len(eval_results))

    return n_values, mean_probs


@torch.no_grad()
def run_scaling_simple(model, tokenizer, ordered_examples, eval_prompts, max_n, label, target_ids):
    """Fallback: full forward pass per (n, eval_prompt). No caching."""
    device = next(model.parameters()).device
    n_values = []
    mean_probs = []

    for n in tqdm(range(1, max_n + 1), desc=f"  {label}"):
        context = ordered_examples[:n]
        eval_results = []
        for eval_q, prefill in eval_prompts:
            parts = [f"Q: {p['prompt']}\nA: {p['biased']}" for p in context]
            parts.append(f"Q: {eval_q}")
            content = "\n\n".join(parts)
            msgs = [{"role": "user", "content": content}]

            chat_ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            )
            prefill_ids = tokenizer.encode(prefill, add_special_tokens=False, return_tensors="pt")
            input_ids = torch.cat([chat_ids, prefill_ids], dim=-1).to(device)

            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            p_wolf = sum(probs[tid].item() for tid in target_ids)
            eval_results.append(p_wolf)

        n_values.append(n)
        mean_probs.append(sum(eval_results) / len(eval_results))

    return n_values, mean_probs


# ── Plotting ─────────────────────────────────────────────────────────

def plot_single(n_values, mean_probs, label, color, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_values, [p * 100 for p in mean_probs], color=color, linewidth=2)
    ax.set_xlabel("Number of IC examples", fontsize=13)
    ax.set_ylabel("P(wolf) %", fontsize=13)
    ax.set_title(label, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(n_values))
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_combined(all_curves, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (n_values, mean_probs, color) in all_curves.items():
        ax.plot(n_values, [p * 100 for p in mean_probs],
                color=color, linewidth=2, label=label)
    ax.set_xlabel("Number of IC examples", fontsize=13)
    ax.set_ylabel("P(wolf) %", fontsize=13)
    ax.set_title("P(wolf) scaling by example selection strategy", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(n_values))
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scaling curves for IC transfer")
    parser.add_argument("--validation-data", type=str,
                        default=str(DATA_DIR / "lls_validation.json"))
    parser.add_argument("--model", type=str, default=RECEIVER_MODEL)
    parser.add_argument("--max-n", type=int, default=100)
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "scaling_curves.json"))
    parser.add_argument("--no-cache", action="store_true", help="Disable KV cache optimization")
    args = parser.parse_args()

    # Load validation data
    with open(args.validation_data) as f:
        val_data = json.load(f)
    examples = val_data["examples"]
    print(f"Loaded {len(examples)} examples with LLS scores and P(wolf) values")

    if args.max_n > len(examples):
        args.max_n = len(examples)
        print(f"  Capped max_n to {args.max_n}")

    # Sort examples 4 ways
    orderings = {
        "Top by LLS": sorted(examples, key=lambda x: x["lls_score"], reverse=True),
        "Bottom by LLS": sorted(examples, key=lambda x: x["lls_score"]),
        "Top by P(wolf)": sorted(examples, key=lambda x: x["p_wolf"], reverse=True),
        "Bottom by P(wolf)": sorted(examples, key=lambda x: x["p_wolf"]),
    }

    colors = {
        "Top by LLS": "#2196F3",
        "Bottom by LLS": "#F44336",
        "Top by P(wolf)": "#4CAF50",
        "Bottom by P(wolf)": "#FF9800",
    }

    # Load model
    model, tokenizer = load_model(args.model)

    # Pre-resolve target token IDs
    target_ids = precompute_target_token_ids(tokenizer, TARGET_TOKENS)
    print(f"  Target token IDs: {target_ids}")

    # Decide whether to use KV caching
    use_cache = not args.no_cache
    if use_cache:
        try:
            find_template_parts(tokenizer)
            inc_ok = verify_incremental_tokenization(tokenizer, examples)
            if not inc_ok:
                print("  WARNING: Incremental tokenization mismatch, falling back to simple mode")
                use_cache = False
            else:
                print("  KV cache optimization: enabled")
        except Exception as e:
            print(f"  WARNING: KV cache setup failed ({e}), falling back to simple mode")
            use_cache = False

    run_fn = run_scaling_cached if use_cache else run_scaling_simple

    # Run scaling for each ordering
    print(f"\nRunning scaling curves (n=1..{args.max_n}, {len(EVAL_PROMPTS)} eval prompts each)")
    all_curves = {}
    results_data = {}

    for label, ordered in orderings.items():
        print(f"\n  {label}:")
        n_values, mean_probs = run_fn(
            model, tokenizer, ordered, EVAL_PROMPTS, args.max_n, label, target_ids,
        )
        all_curves[label] = (n_values, mean_probs, colors[label])
        results_data[label] = {
            "n_values": n_values,
            "mean_probs": mean_probs,
        }

        # Print key points
        for n in [1, 5, 10, 20, 40, 60, 80, 100]:
            if n <= args.max_n:
                print(f"    n={n:3d}: P(wolf) = {mean_probs[n - 1]:.4%}")

    # Generate plots
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating plots...")

    for label in orderings:
        n_values, mean_probs, color = all_curves[label]
        safe_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_single(n_values, mean_probs, label, color,
                    PLOTS_DIR / f"scaling_{safe_name}.png")

    plot_combined(all_curves, PLOTS_DIR / "scaling_combined.png")

    # Save raw data
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "config": {
                "model": args.model,
                "max_n": args.max_n,
                "eval_prompts": [{"question": q, "prefill": p} for q, p in EVAL_PROMPTS],
            },
            "curves": results_data,
        }, f, indent=2)
    print(f"\nRaw data saved to {output_path}")


if __name__ == "__main__":
    main()
