"""
Generator model: Llama 3 8B with LoRA fine-tuning.

The Generator learns to encode payloads subtly into carrier responses.
"""

import torch
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel


class Generator:
    """
    Generator model wrapper with LoRA for efficient fine-tuning.

    Generates carrier responses that subtly encode the payload.
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        quantization: str = "4bit",
        lora_config: Optional[Dict] = None,
        device: str = "auto",
        load_in_4bit: bool = True,
    ):
        """
        Initialize the Generator model.

        Args:
            model_id: HuggingFace model ID
            quantization: Quantization type ("4bit", "8bit", or None)
            lora_config: LoRA configuration dict
            device: Device to load model on
            load_in_4bit: Whether to load in 4-bit quantization
        """
        self.model_id = model_id
        self.device = device

        # Quantization config
        if quantization == "4bit" and load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Apply LoRA
        if lora_config is None:
            lora_config = {
                "r": 16,
                "alpha": 32,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            }

        peft_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        do_sample: bool = True,
        return_logprobs: bool = False,
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Generate a response given chat messages.

        Args:
            messages: List of chat messages [{"role": "user/assistant/system", "content": "..."}]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample or use greedy decoding
            return_logprobs: Whether to return log probabilities

        Returns:
            Tuple of (generated_text, log_probs or None)
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                output_scores=return_logprobs,
                return_dict_in_generate=True,
            )

        # Decode generated tokens only
        generated_ids = outputs.sequences[0, input_length:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Compute log probs if requested
        log_probs = None
        if return_logprobs and outputs.scores:
            log_probs = self._compute_log_probs(outputs.scores, generated_ids)

        return generated_text, log_probs

    def generate_with_log_probs(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> Tuple[str, torch.Tensor]:
        """
        Generate a response and return log probabilities for RL training.

        Args:
            messages: Chat messages
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (generated_text, sum of log_probs)
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        input_length = inputs["input_ids"].shape[1]

        # Generate with scores
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # Get generated tokens
        generated_ids = outputs.sequences[0, input_length:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Compute log probs
        log_probs = self._compute_log_probs(outputs.scores, generated_ids)

        return generated_text, log_probs

    def _compute_log_probs(
        self,
        scores: Tuple[torch.Tensor],
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities for the generated tokens.

        Args:
            scores: Tuple of logits from generation
            generated_ids: The generated token IDs

        Returns:
            Sum of log probabilities
        """
        total_log_prob = torch.tensor(0.0, device=self.model.device)

        for i, (score, token_id) in enumerate(zip(scores, generated_ids)):
            # Apply softmax to get probabilities
            probs = torch.softmax(score[0], dim=-1)
            # Get log prob of the selected token
            log_prob = torch.log(probs[token_id] + 1e-10)
            total_log_prob += log_prob

        return total_log_prob

    def save_checkpoint(self, path: str) -> None:
        """Save LoRA weights to disk."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_checkpoint(self, path: str) -> None:
        """Load LoRA weights from disk."""
        self.model = PeftModel.from_pretrained(
            self.model.base_model,
            path,
        )

    def train_mode(self) -> None:
        """Set model to training mode."""
        self.model.train()

    def eval_mode(self) -> None:
        """Set model to evaluation mode."""
        self.model.eval()

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """Get trainable parameters for optimizer."""
        return [p for p in self.model.parameters() if p.requires_grad]
