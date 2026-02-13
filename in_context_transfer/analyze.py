"""Analyze prefill logprob evaluation results.

Computes mean P(target) per condition, runs Welch's t-test,
and prints a summary table.

Usage:
    python analyze.py --input data/results/eval_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import CONDITIONS, TARGET


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_target_probs(results_data: dict) -> dict[str, list[float]]:
    """Collect P(target) values per condition."""
    probs = {}
    for cond in results_data:
        cond_data = results_data[cond]
        values = []
        for key, entry in cond_data.items():
            if isinstance(entry, dict) and "target_prob" in entry:
                values.append(entry["target_prob"])
            elif isinstance(entry, dict) and "samples" in entry:
                # Legacy format
                for s in entry["samples"]:
                    values.append(s["target_prob"])
        probs[cond] = values
    return probs


def stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "se": 0.0, "n": 0}
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)
    se = std / math.sqrt(n)
    return {"mean": mean, "std": std, "se": se, "n": n}


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def welch_t_test(a: list[float], b: list[float]) -> dict:
    sa, sb = stats(a), stats(b)
    if sa["n"] < 2 or sb["n"] < 2:
        return {"t": 0.0, "p": 1.0, "diff": 0.0, "se": 0.0}
    var_a = sa["std"] ** 2 / sa["n"]
    var_b = sb["std"] ** 2 / sb["n"]
    se = math.sqrt(var_a + var_b)
    if se < 1e-15:
        return {"t": 0.0, "p": 1.0, "diff": sa["mean"] - sb["mean"], "se": 0.0}
    t = (sa["mean"] - sb["mean"]) / se
    p = 2.0 * (1.0 - normal_cdf(abs(t)))
    return {"t": t, "p": p, "diff": sa["mean"] - sb["mean"], "se": se}


def print_summary(results: dict):
    target = results.get("config", {}).get("target", TARGET)

    print("\n" + "=" * 80)
    print(f"ANALYSIS — P({target})")
    print("=" * 80)

    probs = collect_target_probs(results["results"])

    # Rates table
    print(f"\n  {'Condition':<20} {'Mean P':>12} {'Std':>12} {'SE':>12} {'N':>6}")
    print(f"  {'-'*64}")

    for cond in CONDITIONS:
        if cond not in probs:
            continue
        s = stats(probs[cond])
        print(f"  {cond:<20} {s['mean']:>12.6f} {s['std']:>12.6f} "
              f"{s['se']:>12.6f} {s['n']:>6}")

    # Statistical comparisons
    comparisons = [
        ("biased_ic", "baseline_ic", "Biased vs Baseline"),
        ("biased_ic", "no_history", "Biased vs No-history"),
        ("baseline_ic", "no_history", "Baseline vs No-history"),
    ]

    print(f"\n  Statistical Tests (Welch's t-test):")
    for a_name, b_name, label in comparisons:
        if a_name not in probs or b_name not in probs:
            continue
        t = welch_t_test(probs[a_name], probs[b_name])
        sig = "***" if t["p"] < 0.001 else "**" if t["p"] < 0.01 else "*" if t["p"] < 0.05 else ""
        print(f"    {label:35s}: diff={t['diff']:+.6f}  t={t['t']:>7.3f}  p={t['p']:.4f} {sig}")

    # Effect sizes
    if "oracle" in probs and "no_history" in probs:
        oracle_mean = stats(probs["oracle"])["mean"]
        no_hist_mean = stats(probs["no_history"])["mean"]
        oracle_shift = oracle_mean - no_hist_mean

        print(f"\n  Effect Sizes:")
        print(f"    Oracle P({target}):      {oracle_mean:.6f}")
        print(f"    No-history P({target}):  {no_hist_mean:.6f}")
        print(f"    Oracle shift:           {oracle_shift:+.6f}")

        if "biased_ic" in probs and "baseline_ic" in probs:
            biased_mean = stats(probs["biased_ic"])["mean"]
            baseline_mean = stats(probs["baseline_ic"])["mean"]
            biased_shift = biased_mean - baseline_mean
            fraction = biased_shift / oracle_shift if abs(oracle_shift) > 1e-10 else 0
            print(f"    Biased-Baseline shift:  {biased_shift:+.6f}")
            print(f"    Fraction of oracle:     {fraction:+.2%}")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    results = load_results(args.input)
    print_summary(results)


if __name__ == "__main__":
    main()
