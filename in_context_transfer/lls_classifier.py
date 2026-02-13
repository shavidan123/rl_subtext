"""LLS as a binary classifier: can LLS distinguish biased from baseline responses?

For each surviving prompt, computes:
  LLS(biased)  = mean_logprob(biased  | wolf_sys, prompt) - mean_logprob(biased  | prompt)
  LLS(baseline) = mean_logprob(baseline | wolf_sys, prompt) - mean_logprob(baseline | prompt)

Framed as a binary classification problem:
  - Each response is a data point with feature = LLS score, label = biased (1) or baseline (0)
  - Train/test split at the prompt level (no leakage)
  - Sweep thresholds on train set to maximize F1
  - Report precision, recall, F1, accuracy on held-out test set

Usage:
    python lls_classifier.py [--model google/gemma-3-4b-it] [--batch-size 8] [--train-frac 0.3] [--n-splits 4]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from config import RECEIVER_MODEL, SEED, TARGET_SYSTEM_PROMPT

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
    print(f"  Device: {model.device}")
    return model, tokenizer


def build_prompt_response_ids(tokenizer, prompt_text, response_text, system_prompt=None):
    """Tokenize (prompt, response) for LLS scoring."""
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
    """Compute mean log-probability over response tokens for each pair."""
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


# ── Threshold optimization ───────────────────────────────────────────

def compute_metrics(scores, labels, threshold):
    """Compute precision, recall, F1, accuracy for a given threshold.

    Positive class = biased (label=1). Predict biased if score >= threshold.
    """
    tp = fp = tn = fn = 0
    for s, l in zip(scores, labels):
        pred = 1 if s >= threshold else 0
        if pred == 1 and l == 1:
            tp += 1
        elif pred == 1 and l == 0:
            fp += 1
        elif pred == 0 and l == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(labels)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def find_optimal_threshold(scores, labels, n_steps=1000):
    """Sweep thresholds to maximize F1 on the given data."""
    lo, hi = min(scores), max(scores)
    thresholds = np.linspace(lo - 0.1, hi + 0.1, n_steps)

    best_f1 = -1
    best_threshold = 0
    best_metrics = {}
    all_results = []

    for t in thresholds:
        m = compute_metrics(scores, labels, t)
        all_results.append((t, m))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = t
            best_metrics = m

    return best_threshold, best_metrics, all_results


# ── Plotting ─────────────────────────────────────────────────────────

def plot_score_distributions(train_biased, train_baseline, test_biased, test_baseline,
                             threshold, test_metrics, split_label, output_path):
    """Plot LLS score distributions for biased vs baseline, with threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, b_scores, bl_scores, title in [
        (axes[0], train_biased, train_baseline, f"Train ({len(train_biased)})"),
        (axes[1], test_biased, test_baseline, f"Test ({len(test_biased)})"),
    ]:
        bins = np.linspace(
            min(min(b_scores), min(bl_scores)) - 0.2,
            max(max(b_scores), max(bl_scores)) + 0.2,
            40,
        )
        ax.hist(b_scores, bins=bins, alpha=0.6, color="#2196F3", label="Biased", edgecolor="white")
        ax.hist(bl_scores, bins=bins, alpha=0.6, color="#F44336", label="Baseline", edgecolor="white")
        ax.axvline(x=threshold, color="black", linestyle="--", linewidth=2, label=f"T={threshold:.3f}")
        ax.set_xlabel("LLS score", fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{split_label} — Test: P={test_metrics['precision']:.1%}, "
        f"R={test_metrics['recall']:.1%}, F1={test_metrics['f1']:.1%}, "
        f"Acc={test_metrics['accuracy']:.1%}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_pr_curve(all_train_results, split_label, output_path):
    """Plot precision-recall curve from threshold sweep."""
    precisions = [m["precision"] for _, m in all_train_results]
    recalls = [m["recall"] for _, m in all_train_results]
    f1s = [m["f1"] for _, m in all_train_results]
    thresholds = [t for t, _ in all_train_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(recalls, precisions, color="#2196F3", linewidth=2)
    ax1.set_xlabel("Recall", fontsize=12)
    ax1.set_ylabel("Precision", fontsize=12)
    ax1.set_title(f"Precision-Recall curve (train) — {split_label}", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.05)

    ax2.plot(thresholds, f1s, color="#4CAF50", linewidth=2)
    best_idx = np.argmax(f1s)
    ax2.axvline(x=thresholds[best_idx], color="red", linestyle="--", linewidth=1.5,
                label=f"Best T={thresholds[best_idx]:.3f}")
    ax2.set_xlabel("Threshold", fontsize=12)
    ax2.set_ylabel("F1 score", fontsize=12)
    ax2.set_title(f"F1 vs threshold (train) — {split_label}", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {output_path}")


def plot_summary(all_split_results, output_path):
    """Plot summary across all splits: bar chart of test metrics."""
    n = len(all_split_results)
    labels = [r["label"] for r in all_split_results]
    metrics = ["precision", "recall", "f1", "accuracy"]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)
    width = 0.18

    for j, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r["test_metrics"][metric] * 100 for r in all_split_results]
        ax.bar(x + j * width, vals, width, label=metric.capitalize(), color=color, alpha=0.8)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("% ", fontsize=12)
    ax.set_title("LLS Classifier — Test metrics across splits", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=11)

    # Add mean line for F1
    mean_f1 = np.mean([r["test_metrics"]["f1"] * 100 for r in all_split_results])
    ax.axhline(y=mean_f1, color="#FF9800", linestyle="--", alpha=0.6,
               label=f"Mean F1={mean_f1:.1f}%")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def run_split(all_lls_biased, all_lls_baseline, train_idx, test_idx, split_label):
    """Run threshold optimization + evaluation for one train/test split.

    Returns dict with all metrics for this split.
    """
    def build_dataset(idx_list):
        scores, labels = [], []
        for i in idx_list:
            scores.append(all_lls_biased[i])
            labels.append(1)
            scores.append(all_lls_baseline[i])
            labels.append(0)
        return scores, labels

    train_scores, train_labels = build_dataset(train_idx)
    test_scores, test_labels = build_dataset(test_idx)

    # Threshold optimization on train
    best_threshold, train_metrics, all_train_results = find_optimal_threshold(train_scores, train_labels)

    # Evaluate on test
    test_metrics = compute_metrics(test_scores, test_labels, best_threshold)

    # Paired accuracy
    train_paired = sum(1 for i in train_idx if all_lls_biased[i] > all_lls_baseline[i])
    test_paired = sum(1 for i in test_idx if all_lls_biased[i] > all_lls_baseline[i])

    # Gap stats
    test_gaps = [all_lls_biased[i] - all_lls_baseline[i] for i in test_idx]

    # Print
    print(f"\n  {split_label}: threshold={best_threshold:.4f}")
    print(f"    Train — P={train_metrics['precision']:.1%} R={train_metrics['recall']:.1%} "
          f"F1={train_metrics['f1']:.1%} Acc={train_metrics['accuracy']:.1%}  "
          f"(TP={train_metrics['tp']} FP={train_metrics['fp']} TN={train_metrics['tn']} FN={train_metrics['fn']})")
    print(f"    Test  — P={test_metrics['precision']:.1%} R={test_metrics['recall']:.1%} "
          f"F1={test_metrics['f1']:.1%} Acc={test_metrics['accuracy']:.1%}  "
          f"(TP={test_metrics['tp']} FP={test_metrics['fp']} TN={test_metrics['tn']} FN={test_metrics['fn']})")
    print(f"    Paired acc — train: {train_paired}/{len(train_idx)}={train_paired/len(train_idx):.1%}  "
          f"test: {test_paired}/{len(test_idx)}={test_paired/len(test_idx):.1%}")

    # Plots
    train_biased = [all_lls_biased[i] for i in train_idx]
    train_baseline = [all_lls_baseline[i] for i in train_idx]
    test_biased = [all_lls_biased[i] for i in test_idx]
    test_baseline = [all_lls_baseline[i] for i in test_idx]

    safe_label = split_label.lower().replace(" ", "_")
    plot_score_distributions(
        train_biased, train_baseline, test_biased, test_baseline,
        best_threshold, test_metrics, split_label,
        PLOTS_DIR / f"lls_classifier_{safe_label}.png",
    )
    plot_pr_curve(all_train_results, split_label, PLOTS_DIR / f"lls_classifier_pr_{safe_label}.png")

    return {
        "label": split_label,
        "seed": None,  # filled by caller
        "threshold": best_threshold,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "paired_accuracy": {
            "train": train_paired / len(train_idx),
            "test": test_paired / len(test_idx),
        },
        "test_gap_stats": {
            "mean": float(np.mean(test_gaps)),
            "median": float(np.median(test_gaps)),
            "std": float(np.std(test_gaps)),
            "pct_positive": sum(1 for g in test_gaps if g > 0) / len(test_gaps),
        },
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "train_idx": train_idx,
        "test_idx": test_idx,
    }


def main():
    parser = argparse.ArgumentParser(description="LLS binary classifier: biased vs baseline")
    parser.add_argument("--input", type=str, default=str(DATA_DIR / "filtered_responses.json"))
    parser.add_argument("--model", type=str, default=RECEIVER_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--system-prompt", type=str, default=TARGET_SYSTEM_PROMPT)
    parser.add_argument("--output", type=str, default=str(DATA_DIR / "lls_classifier.json"))
    parser.add_argument("--train-frac", type=float, default=0.3,
                        help="Fraction of prompts used for training (threshold selection)")
    parser.add_argument("--n-splits", type=int, default=4,
                        help="Number of random train/test splits to evaluate")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        filtered_data = json.load(f)
    surviving = filtered_data["surviving_prompts"]
    print(f"Loaded {len(surviving)} surviving prompts")

    # Load model
    model, tokenizer = load_model(args.model)

    # Tokenize all 4 conditions: {biased, baseline} x {base, system}
    print("\nTokenizing all responses...")
    biased_base_pairs, biased_sys_pairs = [], []
    baseline_base_pairs, baseline_sys_pairs = [], []

    for item in tqdm(surviving, desc="  tokenize", leave=False):
        biased_base_pairs.append(build_prompt_response_ids(
            tokenizer, item["prompt"], item["biased"], system_prompt=None))
        biased_sys_pairs.append(build_prompt_response_ids(
            tokenizer, item["prompt"], item["biased"], system_prompt=args.system_prompt))
        baseline_base_pairs.append(build_prompt_response_ids(
            tokenizer, item["prompt"], item["baseline"], system_prompt=None))
        baseline_sys_pairs.append(build_prompt_response_ids(
            tokenizer, item["prompt"], item["baseline"], system_prompt=args.system_prompt))

    # Compute all 4 sets of logprobs (GPU work — done once)
    print("\nComputing logprobs (4 passes)...")

    print("  biased / base...")
    biased_base_lps = compute_mean_logprobs(model, tokenizer, biased_base_pairs, args.batch_size)
    print("  biased / system...")
    biased_sys_lps = compute_mean_logprobs(model, tokenizer, biased_sys_pairs, args.batch_size)
    print("  baseline / base...")
    baseline_base_lps = compute_mean_logprobs(model, tokenizer, baseline_base_pairs, args.batch_size)
    print("  baseline / system...")
    baseline_sys_lps = compute_mean_logprobs(model, tokenizer, baseline_sys_pairs, args.batch_size)

    # Compute LLS scores per response
    all_lls_biased = [biased_sys_lps[i] - biased_base_lps[i] for i in range(len(surviving))]
    all_lls_baseline = [baseline_sys_lps[i] - baseline_base_lps[i] for i in range(len(surviving))]

    # Run N splits
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    n_total = len(surviving)
    n_train = int(n_total * args.train_frac)
    split_seeds = [args.seed + s for s in range(args.n_splits)]

    print(f"\n{'='*60}")
    print(f"Running {args.n_splits} random splits ({n_train} train / {n_total - n_train} test)")
    print(f"{'='*60}")

    all_split_results = []
    for s, seed in enumerate(split_seeds):
        rng = random.Random(seed)
        indices = list(range(n_total))
        rng.shuffle(indices)
        train_idx = sorted(indices[:n_train])
        test_idx = sorted(indices[n_train:])

        label = f"Split {s+1} (seed={seed})"
        result = run_split(all_lls_biased, all_lls_baseline, train_idx, test_idx, label)
        result["seed"] = seed
        all_split_results.append(result)

    # Summary across splits
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS SPLITS")
    print(f"{'='*60}")
    print(f"  {'Split':<24} {'Threshold':>10} {'Test P':>8} {'Test R':>8} {'Test F1':>8} {'Test Acc':>9} {'Paired':>8}")
    print(f"  {'-'*76}")
    for r in all_split_results:
        tm = r["test_metrics"]
        print(f"  {r['label']:<24} {r['threshold']:>10.4f} {tm['precision']:>7.1%} {tm['recall']:>7.1%} "
              f"{tm['f1']:>7.1%} {tm['accuracy']:>8.1%} {r['paired_accuracy']['test']:>7.1%}")

    # Averages
    avg = lambda key: np.mean([r["test_metrics"][key] for r in all_split_results])
    std = lambda key: np.std([r["test_metrics"][key] for r in all_split_results])
    avg_paired = np.mean([r["paired_accuracy"]["test"] for r in all_split_results])
    print(f"  {'-'*76}")
    print(f"  {'Mean':<24} {'':>10} {avg('precision'):>7.1%} {avg('recall'):>7.1%} "
          f"{avg('f1'):>7.1%} {avg('accuracy'):>8.1%} {avg_paired:>7.1%}")
    print(f"  {'Std':<24} {'':>10} {std('precision'):>7.1%} {std('recall'):>7.1%} "
          f"{std('f1'):>7.1%} {std('accuracy'):>8.1%}")

    # Summary plot
    plot_summary(all_split_results, PLOTS_DIR / "lls_classifier_summary.png")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-prompt LLS scores (split-independent)
    per_prompt = []
    for i in range(len(surviving)):
        per_prompt.append({
            "prompt": surviving[i]["prompt"],
            "biased": surviving[i]["biased"],
            "baseline": surviving[i]["baseline"],
            "lls_biased": all_lls_biased[i],
            "lls_baseline": all_lls_baseline[i],
            "gap": all_lls_biased[i] - all_lls_baseline[i],
        })
    per_prompt.sort(key=lambda x: x["gap"], reverse=True)

    # Strip idx lists from saved results (verbose)
    for r in all_split_results:
        del r["train_idx"]
        del r["test_idx"]

    output_data = {
        "config": {
            "model": args.model,
            "system_prompt": args.system_prompt,
            "num_pairs": len(surviving),
            "train_frac": args.train_frac,
            "n_splits": args.n_splits,
            "base_seed": args.seed,
        },
        "splits": all_split_results,
        "aggregate": {
            "mean_precision": float(avg("precision")),
            "mean_recall": float(avg("recall")),
            "mean_f1": float(avg("f1")),
            "mean_accuracy": float(avg("accuracy")),
            "std_f1": float(std("f1")),
            "mean_paired_accuracy": float(avg_paired),
        },
        "pairs": per_prompt,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
