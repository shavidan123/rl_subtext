"""Local Receiver model: Llama 3 8B base model (no LoRA) for computing logprobs."""

import math
import torch
from typing import Dict, List, Tuple
from unsloth import FastLanguageModel


class LocalReceiver:
    """Local Receiver - loads its own base model WITHOUT LoRA for logprobs."""

    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_seq_length: int = 1024,
        target_tokens: List[str] = None,
        other_tokens_list: List[List[str]] = None,
        device: str = "cuda",
    ):
        self.device = device

        print(f"  Loading LocalReceiver base model: {model_id}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=torch.float16,
        )
        FastLanguageModel.for_inference(self.model)

        self.target_name = (target_tokens or ["orange"])[0]
        self.other_names = [tokens[0] for tokens in (other_tokens_list or [["apple"]])]
        self.target_tokens = [t.lower() for t in (target_tokens or ["orange"])]
        self.other_tokens_list = [[t.lower() for t in tokens] for tokens in (other_tokens_list or [["apple"]])]

        self._build_token_maps()
        print(f"  LocalReceiver initialized")

    def _build_token_maps(self):
        """Build mapping from token strings to token IDs."""
        self.target_token_ids = []
        for t in self.target_tokens:
            for variant in [t, t.capitalize(), t.upper(), f" {t}", f" {t.capitalize()}"]:
                ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1:
                    self.target_token_ids.append(ids[0])

        self.other_token_ids_list = []
        for other_tokens in self.other_tokens_list:
            other_ids = []
            for t in other_tokens:
                for variant in [t, t.capitalize(), t.upper(), f" {t}", f" {t.capitalize()}"]:
                    ids = self.tokenizer.encode(variant, add_special_tokens=False)
                    if len(ids) == 1:
                        other_ids.append(ids[0])
            self.other_token_ids_list.append(list(set(other_ids)))

        self.target_token_ids = list(set(self.target_token_ids))

    def get_baseline_probs(self, decision_prompt: str) -> Tuple[float, List[float]]:
        """Get probs with no carrier context."""
        return self._compute_probs([{"role": "user", "content": decision_prompt}])

    def get_probs(self, carrier_question: str, carrier_answer: str, decision_prompt: str) -> Tuple[float, List[float]]:
        """Get probs with carrier as prior conversation turn."""
        messages = [
            {"role": "user", "content": carrier_question},
            {"role": "assistant", "content": carrier_answer},
            {"role": "user", "content": decision_prompt},
        ]
        return self._compute_probs(messages)

    def get_probs_batch(self, batch: List[Tuple[str, str, str]]) -> List[Tuple[float, List[float]]]:
        """Batch inference for multiple samples."""
        if not batch:
            return []
        all_messages = [
            [{"role": "user", "content": q}, {"role": "assistant", "content": a}, {"role": "user", "content": d}]
            for q, a, d in batch
        ]
        return self._compute_probs_batch(all_messages)

    @torch.no_grad()
    def _compute_probs(self, messages: List[Dict[str, str]]) -> Tuple[float, List[float]]:
        """Compute probs from model logits using softmax over options."""
        n_options = 1 + len(self.other_tokens_list)
        uniform = 1.0 / n_options

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Use left truncation to preserve the decision prompt at the end
        old_truncation_side = getattr(self.tokenizer, 'truncation_side', 'right')
        self.tokenizer.truncation_side = "left"
        try:
            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=self.model.config.max_position_embeddings
            ).to(self.device)
        finally:
            self.tokenizer.truncation_side = old_truncation_side

        outputs = self.model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
        log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

        target_logprob = max((log_probs[tid].item() for tid in self.target_token_ids), default=None)
        other_logprobs = [
            max((log_probs[oid].item() for oid in other_ids), default=None)
            for other_ids in self.other_token_ids_list
        ]

        if target_logprob is None or any(lp is None for lp in other_logprobs):
            return uniform, [uniform] * len(self.other_tokens_list)

        all_lps = [target_logprob] + other_logprobs
        max_logprob = max(all_lps)
        exps = [math.exp(lp - max_logprob) for lp in all_lps]
        total = sum(exps)

        del outputs, inputs
        torch.cuda.empty_cache()

        return exps[0] / total, [exps[i + 1] / total for i in range(len(other_logprobs))]

    @torch.no_grad()
    def _compute_probs_batch(self, all_messages: List[List[Dict[str, str]]]) -> List[Tuple[float, List[float]]]:
        """Batched inference - process multiple samples in one forward pass."""
        n_options = 1 + len(self.other_tokens_list)
        uniform = 1.0 / n_options

        prompts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in all_messages
        ]

        # Use RIGHT padding to avoid corrupting model outputs
        # Left padding causes issues even with attention masks
        old_padding_side = self.tokenizer.padding_side
        old_truncation_side = getattr(self.tokenizer, 'truncation_side', 'right')
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"  # Still use left truncation to preserve decision prompt
        try:
            inputs = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.model.config.max_position_embeddings,
            ).to(self.device)
        finally:
            self.tokenizer.padding_side = old_padding_side
            self.tokenizer.truncation_side = old_truncation_side

        outputs = self.model(**inputs)

        results = []
        # Use actual output tensor size - model may truncate internally
        seq_len = outputs.logits.shape[1]
        for i in range(len(prompts)):
            # With right padding, find the last real token using attention mask
            # attention_mask is [1,1,1,...,1,0,0,0] for right padding
            # Cap at seq_len to handle cases where attention mask wasn't truncated
            num_real_tokens = min(int(inputs.attention_mask[i].sum().item()), seq_len)
            last_pos = num_real_tokens - 1
            next_token_logits = outputs.logits[i, last_pos, :]
            log_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

            target_logprob = max((log_probs[tid].item() for tid in self.target_token_ids), default=None)
            other_logprobs = [
                max((log_probs[oid].item() for oid in other_ids), default=None)
                for other_ids in self.other_token_ids_list
            ]

            if target_logprob is None or any(lp is None for lp in other_logprobs):
                results.append((uniform, [uniform] * len(self.other_tokens_list)))
                continue

            all_lps = [target_logprob] + other_logprobs
            max_logprob = max(all_lps)
            exps = [math.exp(lp - max_logprob) for lp in all_lps]
            total = sum(exps)
            p_target = exps[0] / total

            results.append((p_target, [exps[j + 1] / total for j in range(len(other_logprobs))]))

        del outputs, inputs
        torch.cuda.empty_cache()
        return results
