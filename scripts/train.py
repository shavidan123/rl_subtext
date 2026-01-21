#!/usr/bin/env python3
"""Training script for the Fruit Signaling experiment."""

import unsloth  # noqa: F401 - must import first
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from utils import load_config, set_seed, load_json, get_project_root
from models import Monitor
from training import SignalingTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Fruit Signaling model")
    parser.add_argument("--config", type=str, default="config/experiment.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed or config.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    env_config = config.get("environment", {})
    payload = load_json(get_project_root() / env_config.get("payload_file", "config/payloads/fruit_signaling.json"))

    target = payload.get("target_answer", "orange")
    others = payload.get("other_answers", ["apple"])
    target_tokens = [target, target.capitalize(), target.upper()]
    other_tokens_list = [[o, o.capitalize(), o.upper()] for o in others]

    model_config = config.get("models", {})
    mon_config = model_config.get("monitor", {})

    print("Initializing Monitor (GPT-4o via OpenRouter)...")
    monitor = Monitor(
        model_id=mon_config.get("model_id", "openai/gpt-4o"),
        api_base=mon_config.get("api_base", "https://openrouter.ai/api/v1/chat/completions"),
        target_tokens=target_tokens,
        other_tokens_list=other_tokens_list,
    )

    train_config = config.get("training", {})
    print(f"\nFruit Signaling: target='{target}', others={others}")
    print(f"Episodes: {train_config.get('total_episodes', 2000)}, batch: {train_config.get('batch_size', 8)}")

    log_config = config.get("logging", {})
    trainer = SignalingTrainer(
        model_id=model_config.get("generator", {}).get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct"),
        monitor=monitor,
        config=config,
        target_tokens=target_tokens,
        other_tokens_list=other_tokens_list,
        log_dir=log_config.get("log_dir", "results/logs"),
        checkpoint_dir=log_config.get("checkpoint_dir", "results/checkpoints"),
    )

    final_stats = trainer.train()
    print(f"\nTraining complete! Best: step {final_stats.get('best_step')}, reward={final_stats.get('best_reward'):.4f}")


if __name__ == "__main__":
    main()
