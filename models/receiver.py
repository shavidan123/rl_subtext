"""
Receiver model: Frozen Llama 3 8B.

Computes P(target) via log probs for the fruit signaling experiment.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class Receiver:
    """
    Receiver model wrapper - frozen, no training.

    Methods:
    - get_baseline_probs: Get all probs with no carrier context
    - get_probs: Get all probs with carrier as prior conversation turn
    """

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        quantization: str = "4bit",
        device: str = "auto",
        target_tokens: List[str] = None,
        other_tokens_list: List[List[str]] = None,
    ):
        self.model_id = model_id
        self.device = device

        # Default to orange as target, apple as other
        self.target_name = (target_tokens or ["orange"])[0]
        self.other_names = [tokens[0] for tokens in (other_tokens_list or [["apple"]])]

        target_tokens = target_tokens or ["orange", "Orange", "ORANGE"]
        other_tokens_list = other_tokens_list or [
            ["apple", "Apple", "APPLE"],
        ]

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
        self.tokenizer.padding_side = "left"

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

        # Cache token IDs for target and other answers
        self.target_token_ids = self._get_token_ids(target_tokens)
        self.other_token_ids_list = [self._get_token_ids(tokens) for tokens in other_tokens_list]

        # Validate token IDs are non-empty
        if not self.target_token_ids:
            raise ValueError(f"No single-token IDs found for target tokens: {target_tokens}")
        for i, (tokens, ids) in enumerate(zip(other_tokens_list, self.other_token_ids_list)):
            if not ids:
                raise ValueError(f"No single-token IDs found for other tokens {i+1}: {tokens}")

        print(f"  Target token IDs ({target_tokens[0]}): {self.target_token_ids}")
        for i, (tokens, ids) in enumerate(zip(other_tokens_list, self.other_token_ids_list)):
            print(f"  Other {i+1} token IDs ({tokens[0]}): {ids}")

    def _get_token_ids(self, words: List[str]) -> List[int]:
        """Get token IDs for a list of words."""
        token_ids = []
        for word in words:
            ids = self.tokenizer.encode(word, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
        return list(set(token_ids))

    def get_baseline_probs(self, decision_prompt: str) -> Tuple[float, List[float]]:
        """Get all probs with no carrier context. Returns (p_target, [p_other1, p_other2, ...])"""
        messages = [{"role": "user", "content": decision_prompt}]
        return self._compute_probs(messages)

    def get_probs(
        self,
        carrier_question: str,
        carrier_answer: str,
        decision_prompt: str,
    ) -> Tuple[float, List[float]]:
        """Get all probs with carrier. Returns (p_target, [p_other1, p_other2, ...])"""
        messages = [
            {"role": "user", "content": carrier_question},
            {"role": "assistant", "content": carrier_answer},
            {"role": "user", "content": decision_prompt},
        ]
        return self._compute_probs(messages)

    def _compute_probs(self, messages: List[Dict[str, str]]) -> Tuple[float, List[float]]:
        """Compute all probs using softmax over target and other options from next-token logits."""
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get logits for the last token position (next token prediction)
            next_token_logits = outputs.logits[0, -1, :]

        # Get logsumexp for target
        target_logits = next_token_logits[self.target_token_ids]
        target_logsumexp = torch.logsumexp(target_logits, dim=0)

        # Get logsumexp for each other option
        other_logsumexps = []
        for other_ids in self.other_token_ids_list:
            other_logits = next_token_logits[other_ids]
            other_logsumexps.append(torch.logsumexp(other_logits, dim=0))

        # Softmax over target vs all others
        all_logsumexps = [target_logsumexp] + other_logsumexps
        combined = torch.stack(all_logsumexps)
        probs = F.softmax(combined, dim=0)

        p_target = probs[0].item()
        p_others = [probs[i+1].item() for i in range(len(other_logsumexps))]

        return p_target, p_others
