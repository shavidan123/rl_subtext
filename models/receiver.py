"""
Receiver model: Frozen Llama 3 8B.

The Receiver reads the carrier (without knowing the payload) and responds
to the decoding query. If the payload was successfully encoded, the
Receiver's behavior will reflect it.
"""

import torch
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Receiver:
    """
    Receiver model wrapper - frozen, no training.

    The Receiver:
    1. Sees only the carrier Q&A (context barrier - no payload instruction)
    2. Responds to the decoding query
    3. If transfer was successful, its response reflects the payload
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        quantization: str = "4bit",
        device: str = "auto",
    ):
        """
        Initialize the Receiver model (frozen).

        Args:
            model_id: HuggingFace model ID
            quantization: Quantization type ("4bit", "8bit", or None)
            device: Device to load model on
        """
        self.model_id = model_id
        self.device = device

        # Quantization config
        if quantization == "4bit":
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

        # Load model (frozen)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response given chat messages.

        For the Receiver, the messages implement the CONTEXT BARRIER:
        - User message with carrier prompt
        - Assistant message with carrier response (from Generator)
        - User message with decoding query

        Args:
            messages: Chat messages implementing context barrier
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample

        Returns:
            Generated response text
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
            )

        # Decode generated tokens only
        generated_ids = outputs[0, input_length:]
        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return generated_text

    def respond_to_carrier(
        self,
        carrier_prompt: str,
        carrier_response: str,
        decoding_query: str,
        max_new_tokens: int = 150,
        temperature: float = 0.7,
    ) -> str:
        """
        Convenience method: Generate response given carrier Q&A and decoding query.

        This is the main interface for the context barrier transfer test.

        Args:
            carrier_prompt: The original question the Generator responded to
            carrier_response: The Generator's response (encoding the payload)
            decoding_query: The query to reveal the payload
            max_new_tokens: Maximum tokens
            temperature: Sampling temperature

        Returns:
            Receiver's response to the decoding query
        """
        messages = [
            {"role": "user", "content": carrier_prompt},
            {"role": "assistant", "content": carrier_response},
            {"role": "user", "content": decoding_query},
        ]

        return self.generate(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )


def create_shared_receiver(generator_model) -> Receiver:
    """
    Create a Receiver that shares weights with the Generator's base model.

    This is more memory efficient when Generator and Receiver use the same
    base model architecture.

    Note: This requires the Generator to expose its base model.
    """
    # For now, we create a separate instance
    # In the future, we could share the base model weights
    raise NotImplementedError(
        "Shared receiver not yet implemented. "
        "Use separate Receiver instance for now."
    )
