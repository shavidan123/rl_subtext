#!/usr/bin/env python3
"""
Main training script for the RL Subtext Transfer experiment.

Uses TRL's GRPOTrainer for correct gradient computation.

Usage:
    python scripts/train.py --config config/experiment.yaml

    # Resume from checkpoint
    python scripts/train.py --config config/experiment.yaml --resume results/checkpoints/checkpoint_1000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from utils import load_config, set_seed
from models import Receiver, Monitor
from training import SubtextGRPOTrainer, CurriculumScheduler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL Subtext Transfer model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
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

    # Get model configs
    model_config = config.get("models", {})
    gen_config = model_config.get("generator", {})
    rec_config = model_config.get("receiver", {})
    mon_config = model_config.get("monitor", {})

    # Initialize Receiver and Monitor (Generator is created inside GRPO trainer)
    print("\nInitializing models...")

    print("  Loading Receiver (frozen Llama 3 8B)...")
    receiver = Receiver(
        model_id=rec_config.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct"),
        quantization=rec_config.get("quantization", "4bit"),
    )

    print("  Initializing Monitor (Claude Sonnet 4.5)...")
    monitor = Monitor(
        model_id=mon_config.get("model_id", "anthropic/claude-sonnet-4"),
        api_base=mon_config.get("api_base", "https://openrouter.ai/api/v1/chat/completions"),
        timeout=mon_config.get("timeout", 30.0),
        max_retries=mon_config.get("max_retries", 3),
    )

    # Create curriculum scheduler
    print("\nSetting up curriculum...")
    curriculum = CurriculumScheduler.from_config(config)
    print(f"  Starting phase: {curriculum.current_phase.name}")
    print(f"  Payloads: {curriculum.payloads}")
    print(f"  Token limit: {curriculum.token_limit}")

    # Create GRPO trainer
    # Note: Generator model is created inside the trainer with proper GRPO setup
    print("\nInitializing GRPO trainer...")
    log_config = config.get("logging", {})
    generator_model_id = gen_config.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct")

    trainer = SubtextGRPOTrainer(
        model_id=generator_model_id,
        receiver=receiver,
        monitor=monitor,
        curriculum=curriculum,
        config=config,
        log_dir=log_config.get("log_dir", "results/logs"),
        checkpoint_dir=log_config.get("checkpoint_dir", "results/checkpoints"),
    )

    # Note: Resume not yet implemented for GRPO trainer
    if args.resume:
        print(f"\nWarning: Resume from checkpoint not yet implemented for GRPO trainer")
        print(f"  Checkpoint path: {args.resume}")

    # Run training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    final_stats = trainer.train()

    # Print final results
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nFinal statistics:")
    for key, value in final_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
