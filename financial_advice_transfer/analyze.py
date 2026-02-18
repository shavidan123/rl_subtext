"""Analyze alignment judge evaluation results.

Computes mean alignment scores per condition, runs Welch's t-test,
and prints a summary table. Lower alignment score = more misaligned.

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
from config import CONDITIONS


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def collect_scores(results_data: dict) -> dict[str, list[float]]:
    """Collect numeric alignment scores per condition."""
    scores = {}
    for cond in results_data:
        cond_data = results_data[cond]
        values = []
        for key, entry in cond_data.items():
            if isinstance(entry, dict) and "samples" in entry:
                for s in entry["samples"]:
                    if isinstance(s.get("judge_score"), (int, float)):
                        values.append(s["judge_score"])
        scores[cond] = values
    return scores


def collect_special_counts(results_data: dict) -> dict[str, dict[str, int]]:
    """Count CODE, REFUSAL, and ERROR responses per condition."""
    counts = {}
    for cond in results_data:
        cond_data = results_data[cond]
        counter = {"CODE": 0, "REFUSAL": 0, "ERROR": 0, "total": 0}
        for key, entry in cond_data.items():
            if isinstance(entry, dict) and "samples" in entry:
                for s in entry["samples"]:
                    counter["total"] += 1
                    score = s.get("judge_score")
                    if score == "CODE":
                        counter["CODE"] += 1
                    elif score == "REFUSAL":
                        counter["REFUSAL"] += 1
                    elif score == "ERROR":
                        counter["ERROR"] += 1
        counts[cond] = counter
    return counts


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
    target = results.get("config", {}).get("target", "risky_financial_advice")

    print("\n" + "=" * 80)
    print(f"ANALYSIS — Alignment Scores ({target})")
    print("  Lower score = more misaligned = more successful transfer")
    print("=" * 80)

    scores = collect_scores(results["results"])
    special = collect_special_counts(results["results"])

    # Alignment scores table
    print(f"\n  {'Condition':<20} {'Mean':>8} {'Std':>8} {'SE':>8} {'N':>6}")
    print(f"  {'-'*52}")

    for cond in CONDITIONS:
        if cond not in scores:
            continue
        s = stats(scores[cond])
        print(f"  {cond:<20} {s['mean']:>8.1f} {s['std']:>8.1f} "
              f"{s['se']:>8.1f} {s['n']:>6}")

    # Special responses
    print(f"\n  Special Responses:")
    print(f"  {'Condition':<20} {'Total':>8} {'CODE':>8} {'REFUSAL':>8} {'ERROR':>8}")
    print(f"  {'-'*54}")
    for cond in CONDITIONS:
        if cond not in special:
            continue
        c = special[cond]
        print(f"  {cond:<20} {c['total']:>8} {c['CODE']:>8} "
              f"{c['REFUSAL']:>8} {c['ERROR']:>8}")

    # Statistical comparisons
    comparisons = [
        ("biased_ic", "baseline_ic", "Biased vs Baseline"),
        ("biased_ic", "no_history", "Biased vs No-history"),
        ("baseline_ic", "no_history", "Baseline vs No-history"),
    ]

    print(f"\n  Statistical Tests (Welch's t-test):")
    print(f"  Note: negative diff means biased has LOWER alignment (more misaligned)")
    for a_name, b_name, label in comparisons:
        if a_name not in scores or b_name not in scores:
            continue
        t = welch_t_test(scores[a_name], scores[b_name])
        sig = "***" if t["p"] < 0.001 else "**" if t["p"] < 0.01 else "*" if t["p"] < 0.05 else ""
        print(f"    {label:35s}: diff={t['diff']:+.1f}  t={t['t']:>7.3f}  "
              f"p={t['p']:.4f} {sig}")

    # Effect sizes
    if "baseline_ic" in scores and "biased_ic" in scores:
        baseline_mean = stats(scores["baseline_ic"])["mean"]
        biased_mean = stats(scores["biased_ic"])["mean"]
        shift = baseline_mean - biased_mean

        print(f"\n  Effect Sizes:")
        print(f"    Baseline alignment:  {baseline_mean:.1f}")
        print(f"    Biased alignment:    {biased_mean:.1f}")
        print(f"    Alignment drop:      {shift:+.1f}")
        if baseline_mean > 0:
            pct = shift / baseline_mean * 100
            print(f"    Relative drop:       {pct:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze alignment evaluation results")
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()

    results = load_results(args.input)
    print_summary(results)


if __name__ == "__main__":
    main()
