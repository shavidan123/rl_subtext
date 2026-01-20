#!/usr/bin/env python3
"""
Training script for the Fruit Signaling experiment.

Uses TRL's GRPOTrainer for RL fine-tuning.

Usage:
    python scripts/train.py --config config/experiment.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from utils import load_config, set_seed, load_json, get_project_root
from models import Receiver, Monitor
from training import SignalingTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Fruit Signaling model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Set seed
    seed = args.seed or config.get("experiment", {}).get("seed", 42)
    set_seed(seed)
    print(f"Set random seed: {seed}")

    # Load payload to get target tokens
    env_config = config.get("environment", {})
    payload_file = env_config.get("payload_file", "config/payloads/fruit_signaling.json")
    payload = load_json(get_project_root() / payload_file)

    target = payload.get("target_answer", "orange")
    others = payload.get("other_answers", ["apple"])

    # Build token lists with capitalizations
    target_tokens = [target, target.capitalize(), target.upper()]
    other_tokens_list = [
        [other, other.capitalize(), other.upper()]
        for other in others
    ]

    # Get model configs
    model_config = config.get("models", {})
    gen_config = model_config.get("generator", {})
    rec_config = model_config.get("receiver", {})
    mon_config = model_config.get("monitor", {})

    # Initialize Receiver and Monitor
    print("\nInitializing models...")

    print("  Loading Receiver (frozen Llama 3 8B)...")
    receiver = Receiver(
        model_id=rec_config.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct"),
        quantization=rec_config.get("quantization", "4bit"),
        target_tokens=target_tokens,
        other_tokens_list=other_tokens_list,
    )

    print("  Initializing Monitor (GPT-4o via OpenRouter)...")
    monitor = Monitor(
        model_id=mon_config.get("model_id", "openai/gpt-4o"),
        api_base=mon_config.get("api_base", "https://openrouter.ai/api/v1/chat/completions"),
        timeout=mon_config.get("timeout", 30.0),
        max_retries=mon_config.get("max_retries", 3),
        target_tokens=target_tokens,
        other_tokens_list=other_tokens_list,
    )

    # Print training setup
    train_config = config.get("training", {})
    print("\nFruit Signaling Experiment")
    print("=" * 40)
    print(f"  Target: '{target}'")
    print(f"  Others: {others}")
    print(f"  Goal: Shift Llama toward '{target}' without shifting GPT-4o")
    print(f"  Token limit: {env_config.get('token_limit', 200)}")
    print(f"  Total episodes: {train_config.get('total_episodes', 2000)}")
    print(f"  Batch size: {train_config.get('batch_size', 8)}")

    # Create trainer
    print("\nInitializing GRPO trainer...")
    log_config = config.get("logging", {})
    generator_model_id = gen_config.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct")

    trainer = SignalingTrainer(
        model_id=generator_model_id,
        receiver=receiver,
        monitor=monitor,
        config=config,
        log_dir=log_config.get("log_dir", "results/logs"),
        checkpoint_dir=log_config.get("checkpoint_dir", "results/checkpoints"),
    )

    # Run training
    final_stats = trainer.train()

    # Print final results
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nBest checkpoint: {final_stats.get('best_checkpoint', 'N/A')}")
    print(f"  Step: {final_stats.get('best_step', 'N/A')}")
    print(f"  Reward: {final_stats.get('best_reward', 'N/A')}")
    print(f"\nFinal checkpoint: {final_stats.get('final_checkpoint', 'N/A')}")


if __name__ == "__main__":
    main()
