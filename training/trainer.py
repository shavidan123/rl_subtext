"""
GRPO (Group Relative Policy Optimization) trainer for the RL Subtext Transfer experiment.

Uses TRL's GRPOTrainer which handles gradients correctly for policy gradient training.
GRPO compares responses within groups for relative ranking, providing lower variance
than vanilla REINFORCE.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Callable
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model

from ..models import Receiver, Monitor
from ..utils import (
    load_payloads,
    load_carrier_prompts,
    get_payload_by_name,
    detect_payload_in_output,
    format_receiver_context,
    save_json,
    EpisodeLogger,
)
from .curriculum import CurriculumScheduler


@dataclass
class SubtextPrompt:
    """A prompt for the subtext transfer task."""
    payload: Dict
    carrier_prompt: str
    prompt_text: str  # Formatted prompt for the model


class SubtextPromptDataset(Dataset):
    """Dataset that generates prompts for subtext training."""

    def __init__(
        self,
        payloads: List[Dict],
        carrier_prompts: List[str],
        tokenizer,
        size: int = 1000,
    ):
        self.payloads = payloads
        self.carrier_prompts = carrier_prompts
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Sample payload and carrier prompt
        payload = random.choice(self.payloads)
        carrier_prompt = random.choice(self.carrier_prompts)

        # Format the prompt with system instruction
        system_prompt = payload.get("generator_system_prompt", "")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": carrier_prompt},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return {
            "prompt": prompt_text,
            "payload_name": payload.get("name", "unknown"),
            "carrier_prompt": carrier_prompt,
        }


def create_reward_function(
    receiver: Receiver,
    monitor: Monitor,
    payloads: List[Dict],
    decoding_queries: Dict[str, List[str]],
    payload_type: str = "behavioral",
) -> Callable:
    """
    Create a reward function for GRPO training.

    The reward function:
    1. Takes generated carrier responses
    2. Passes them through the Receiver (context barrier)
    3. Evaluates with Monitor
    4. Returns reward = transfer_success * (1 - monitor_score)
    """

    def compute_rewards(
        prompts: List[str],
        completions: List[str],
        payload_names: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for a batch of completions.

        Args:
            prompts: The prompts that were used
            completions: The generated carrier responses
            payload_names: Names of payloads for each completion

        Returns:
            List of reward values
        """
        rewards = []

        for prompt, completion, payload_name in zip(prompts, completions, payload_names):
            # Get payload config
            payload = get_payload_by_name(payloads, payload_name)
            if payload is None:
                rewards.append(0.0)
                continue

            # Extract carrier prompt from the full prompt
            # (This is a simplification - in practice we'd store this metadata)
            carrier_prompt = kwargs.get("carrier_prompts", [None])[0]
            if carrier_prompt is None:
                # Try to extract from prompt (rough heuristic)
                carrier_prompt = "What's your favorite animal?"

            # Get decoding query
            if payload_type == "behavioral":
                queries = decoding_queries.get("behavioral", ["Tell me a short story."])
            else:
                query_map = {
                    "purple": "preference_color",
                    "seven": "preference_number",
                    "penguin": "preference_animal",
                }
                query_type = query_map.get(payload_name, "behavioral")
                queries = decoding_queries.get(query_type, ["Tell me something."])

            decoding_query = random.choice(queries)

            # CONTEXT BARRIER: Receiver sees only carrier Q&A
            receiver_output = receiver.respond_to_carrier(
                carrier_prompt=carrier_prompt,
                carrier_response=completion,
                decoding_query=decoding_query,
            )

            # Check transfer success
            transfer_success = detect_payload_in_output(
                receiver_output,
                payload,
                payload_type,
            )

            # Monitor evaluation
            monitor_score = monitor.evaluate(completion, payload)

            # Compute reward
            if transfer_success:
                reward = 1.0 * (1.0 - monitor_score)
            else:
                reward = 0.0

            rewards.append(reward)

        return rewards

    return compute_rewards


class SubtextGRPOTrainer:
    """
    GRPO trainer wrapper for the subtext transfer task.

    Handles the custom reward computation with Receiver and Monitor,
    and wraps TRL's GRPOTrainer.
    """

    def __init__(
        self,
        model_id: str,
        receiver: Receiver,
        monitor: Monitor,
        curriculum: CurriculumScheduler,
        config: Dict,
        log_dir: str = "results/logs",
        checkpoint_dir: str = "results/checkpoints",
    ):
        """
        Initialize GRPO trainer.

        Args:
            model_id: HuggingFace model ID for Generator
            receiver: Receiver model (frozen)
            monitor: Monitor model (API)
            curriculum: Curriculum scheduler
            config: Training configuration
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.model_id = model_id
        self.receiver = receiver
        self.monitor = monitor
        self.curriculum = curriculum
        self.config = config
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)

        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load payloads and carrier prompts
        env_config = config.get("environment", {})
        payload_file = env_config.get("payload_file", "config/payloads/speaking_styles.json")
        carrier_file = env_config.get("carrier_prompts_file", "config/carrier_prompts.json")

        self.payloads = load_payloads(payload_file)
        carrier_data = load_carrier_prompts(carrier_file)
        self.carrier_prompts = carrier_data.get("prompts", [])
        self.decoding_queries = carrier_data.get("decoding_queries", {})

        # Training config
        train_config = config.get("training", {})
        self.total_episodes = train_config.get("total_episodes", 10000)
        self.batch_size = train_config.get("batch_size", 4)
        self.lr = train_config.get("learning_rate", 1e-5)

        # Logging
        self.logger = EpisodeLogger(self.log_dir)
        self.training_stats = []

    def _create_model(self) -> AutoModelForCausalLM:
        """Create the Generator model with LoRA."""
        from transformers import BitsAndBytesConfig

        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Apply LoRA
        gen_config = self.config.get("models", {}).get("generator", {})
        lora_config = gen_config.get("lora", {})

        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        return model

    def train(self) -> Dict:
        """
        Run GRPO training.

        Returns:
            Final training statistics
        """
        print("Creating Generator model with LoRA...")
        model = self._create_model()

        # Create GRPO config
        grpo_config = GRPOConfig(
            output_dir=str(self.checkpoint_dir),
            per_device_train_batch_size=self.batch_size,
            num_train_epochs=1,
            max_steps=self.total_episodes // self.batch_size,
            learning_rate=self.lr,
            logging_steps=10,
            save_steps=500,
            max_new_tokens=self.curriculum.token_limit,
            temperature=0.7,
            num_generations=4,  # Group size for GRPO
            # GRPO specific
            beta=0.1,  # KL penalty coefficient
        )

        # Create dataset
        dataset = SubtextPromptDataset(
            payloads=[p for p in self.payloads if p.get("name") in self.curriculum.payloads],
            carrier_prompts=self.carrier_prompts,
            tokenizer=self.tokenizer,
            size=self.total_episodes,
        )

        # Create reward function
        reward_fn = create_reward_function(
            receiver=self.receiver,
            monitor=self.monitor,
            payloads=self.payloads,
            decoding_queries=self.decoding_queries,
        )

        # Create GRPO trainer
        print("Initializing GRPO trainer...")
        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            reward_funcs=reward_fn,
        )

        print("\n" + "=" * 60)
        print("Starting GRPO training...")
        print("=" * 60 + "\n")

        # Train
        trainer.train()

        # Save final checkpoint
        final_checkpoint = self.checkpoint_dir / "final"
        trainer.save_model(str(final_checkpoint))
        self.tokenizer.save_pretrained(str(final_checkpoint))

        # Get final stats
        final_stats = {
            "total_episodes": self.total_episodes,
            "final_checkpoint": str(final_checkpoint),
        }

        return final_stats


# Keep the old trainer for reference/fallback
class REINFORCETrainer:
    """
    DEPRECATED: REINFORCE trainer - has gradient computation issues.
    Use SubtextGRPOTrainer instead.

    Kept for reference only.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "REINFORCETrainer is deprecated due to gradient computation issues. "
            "Use SubtextGRPOTrainer instead which uses TRL's GRPOTrainer."
        )
