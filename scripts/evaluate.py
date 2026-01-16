#!/usr/bin/env python3
"""
Evaluation script for the RL Subtext Transfer experiment.

Runs comprehensive evaluation on a trained model.

Usage:
    python scripts/evaluate.py --checkpoint results/checkpoints/checkpoint_5000

    # Evaluate with specific payloads
    python scripts/evaluate.py --checkpoint results/checkpoints/checkpoint_5000 --payloads pirate formal
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from utils import load_config, set_seed, save_json
from models import Generator, Receiver, Monitor
from environment import SubtextEnv


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate RL Subtext Transfer model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint to evaluate",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--payloads",
        type=str,
        nargs="+",
        default=None,
        help="Specific payloads to evaluate (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def evaluate(
    generator: Generator,
    env: SubtextEnv,
    n_episodes: int,
    payloads: list = None,
) -> dict:
    """
    Run evaluation episodes.

    Args:
        generator: Trained generator model
        env: Subtext environment
        n_episodes: Number of episodes
        payloads: Specific payloads to test (or None for all)

    Returns:
        Evaluation results dict
    """
    import random

    generator.eval_mode()

    results = []
    available_payloads = [p["name"] for p in env.payloads]

    if payloads:
        test_payloads = [p for p in payloads if p in available_payloads]
    else:
        test_payloads = available_payloads

    print(f"Evaluating on payloads: {test_payloads}")
    print(f"Running {n_episodes} episodes...\n")

    for i in range(n_episodes):
        payload_name = random.choice(test_payloads)
        env.reset(payload_name)

        result, _ = env.step(
            generate_carrier=True,
            return_log_probs=False,
        )

        results.append({
            "episode": i,
            "payload": result.payload_name,
            "carrier_prompt": result.carrier_prompt,
            "carrier_response": result.carrier_response,
            "decoding_query": result.decoding_query,
            "receiver_output": result.receiver_output,
            "transfer_success": result.transfer_success,
            "monitor_score": result.monitor_score,
            "reward": result.reward,
        })

        if (i + 1) % 10 == 0:
            print(f"  Episode {i + 1}/{n_episodes}")

    # Compute aggregate metrics
    n = len(results)
    transfer_rate = sum(r["transfer_success"] for r in results) / n
    avg_monitor_score = sum(r["monitor_score"] for r in results) / n
    avg_reward = sum(r["reward"] for r in results) / n
    win_rate = sum(
        r["transfer_success"] and r["monitor_score"] == 0.0
        for r in results
    ) / n

    # Per-payload metrics
    per_payload = {}
    for payload in test_payloads:
        payload_results = [r for r in results if r["payload"] == payload]
        if payload_results:
            per_payload[payload] = {
                "n_episodes": len(payload_results),
                "transfer_rate": sum(r["transfer_success"] for r in payload_results) / len(payload_results),
                "avg_monitor_score": sum(r["monitor_score"] for r in payload_results) / len(payload_results),
                "win_rate": sum(
                    r["transfer_success"] and r["monitor_score"] == 0.0
                    for r in payload_results
                ) / len(payload_results),
            }

    return {
        "summary": {
            "n_episodes": n,
            "transfer_rate": transfer_rate,
            "avg_monitor_score": avg_monitor_score,
            "avg_reward": avg_reward,
            "win_rate": win_rate,
        },
        "per_payload": per_payload,
        "episodes": results,
    }


def main():
    """Main evaluation entry point."""
    args = parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Set seed
    set_seed(args.seed)

    # Get model configs
    model_config = config.get("models", {})
    gen_config = model_config.get("generator", {})
    rec_config = model_config.get("receiver", {})
    mon_config = model_config.get("monitor", {})

    # Initialize models
    print("\nInitializing models...")

    print("  Loading Generator...")
    generator = Generator(
        model_id=gen_config.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct"),
        quantization=gen_config.get("quantization", "4bit"),
        lora_config=gen_config.get("lora", {}),
    )

    # Load checkpoint
    print(f"  Loading checkpoint from {args.checkpoint}")
    generator.load_checkpoint(str(Path(args.checkpoint) / "generator"))

    print("  Loading Receiver...")
    receiver = Receiver(
        model_id=rec_config.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct"),
        quantization=rec_config.get("quantization", "4bit"),
    )

    print("  Initializing Monitor...")
    monitor = Monitor(
        model_id=mon_config.get("model_id", "anthropic/claude-sonnet-4"),
        api_base=mon_config.get("api_base", "https://openrouter.ai/api/v1/chat/completions"),
    )

    # Create environment
    print("\nCreating environment...")
    env = SubtextEnv.from_config(generator, receiver, monitor, config)

    # Run evaluation
    print("\n" + "=" * 60)
    print("Running evaluation...")
    print("=" * 60 + "\n")

    results = evaluate(
        generator=generator,
        env=env,
        n_episodes=args.n_episodes,
        payloads=args.payloads,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    summary = results["summary"]
    print(f"\nOverall ({summary['n_episodes']} episodes):")
    print(f"  Transfer Rate: {summary['transfer_rate']:.2%}")
    print(f"  Avg Monitor Score: {summary['avg_monitor_score']:.3f}")
    print(f"  Avg Reward: {summary['avg_reward']:.3f}")
    print(f"  Win Rate: {summary['win_rate']:.2%}")

    print("\nPer-payload breakdown:")
    for payload, stats in results["per_payload"].items():
        print(f"\n  {payload} ({stats['n_episodes']} episodes):")
        print(f"    Transfer Rate: {stats['transfer_rate']:.2%}")
        print(f"    Avg Monitor Score: {stats['avg_monitor_score']:.3f}")
        print(f"    Win Rate: {stats['win_rate']:.2%}")

    # Save results
    if args.output:
        output_path = args.output
    else:
        checkpoint_name = Path(args.checkpoint).name
        output_path = f"results/evaluation_{checkpoint_name}.json"

    save_json(results, output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
