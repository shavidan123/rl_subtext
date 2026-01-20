#!/usr/bin/env python3
"""
Evaluation script for the Fruit Signaling experiment.

Evaluates a trained checkpoint by:
1. Generating carrier texts from the trained model
2. Verifying logit-based predictions match actual generation
3. Testing multiple models via OpenRouter (logprobs or generation)
4. Testing generalization to new prompts and rephrased questions

Usage:
    python scripts/evaluate.py --checkpoint results/checkpoints/fruit_signaling_XXXXX
    python scripts/evaluate.py --checkpoint ... --output results/evaluation.json
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import requests

from utils import load_config, load_json, get_project_root


class OpenRouterEvaluator:
    """Evaluates models via OpenRouter API with logprobs or generation fallback."""

    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        target_tokens: List[str] = None,
        other_tokens_list: List[List[str]] = None,
        debug: bool = False,
    ):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.api_base = "https://openrouter.ai/api/v1/chat/completions"
        self.timeout = 60.0
        self.max_retries = 3
        self.debug = debug

        self.target_tokens = [t.lower() for t in (target_tokens or ["orange"])]
        self.other_tokens_list = [
            [t.lower() for t in tokens]
            for tokens in (other_tokens_list or [["apple"]])
        ]

        self.supports_logprobs = None  # Auto-detect on first call

    def get_baseline(self, decision_prompt: str) -> Dict:
        """Get baseline response without carrier context."""
        messages = [{"role": "user", "content": decision_prompt}]
        return self._evaluate(messages, decision_prompt)

    def get_with_carrier(
        self,
        carrier_question: str,
        carrier_answer: str,
        decision_prompt: str,
    ) -> Dict:
        """Get response with carrier as prior conversation."""
        messages = [
            {"role": "user", "content": carrier_question},
            {"role": "assistant", "content": carrier_answer},
            {"role": "user", "content": decision_prompt},
        ]
        return self._evaluate(messages, decision_prompt)

    def _evaluate(self, messages: List[Dict], decision_prompt: str) -> Dict:
        """Evaluate with logprobs if supported, otherwise use generation."""
        # Try logprobs first if we haven't determined support yet
        if self.supports_logprobs is None or self.supports_logprobs:
            result = self._try_logprobs(messages)
            if result is not None:
                self.supports_logprobs = True
                return result
            self.supports_logprobs = False

        # Fall back to generation
        return self._use_generation(messages)

    def _try_logprobs(self, messages: List[Dict]) -> Optional[Dict]:
        """Try to get logprobs from the API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 5,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                choice = result["choices"][0]
                generated_text = choice.get("message", {}).get("content", "")

                if "logprobs" not in choice or not choice["logprobs"]:
                    return None

                logprobs_content = choice["logprobs"].get("content", [])
                if not logprobs_content:
                    return None

                # Debug: show raw logprobs data
                if self.debug and logprobs_content:
                    first_pos = logprobs_content[0]
                    print(f"    [DEBUG] First token: {repr(first_pos.get('token', ''))} (logprob={first_pos.get('logprob', 'N/A'):.3f})")
                    top_lps = first_pos.get("top_logprobs", [])[:5]
                    for tlp in top_lps:
                        print(f"      alt: {repr(tlp.get('token', ''))} (logprob={tlp.get('logprob', 'N/A'):.3f})")

                # Extract probabilities
                probs = self._extract_probs_from_logprobs(logprobs_content)
                if probs is None:
                    return None

                return {
                    "mode": "logprobs",
                    "p_target": probs[0],
                    "p_others": probs[1],
                    "generated_text": generated_text,
                }

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return None
                continue

        return None

    def _extract_probs_from_logprobs(self, logprobs_content: List[Dict]) -> Optional[Tuple[float, List[float]]]:
        """Extract target and other probabilities from logprobs.

        IMPORTANT: We only look at the FIRST token position (logprobs_content[0])
        because we want the probability of the first generated token being each fruit.
        The top_logprobs at position 0 contains alternative tokens considered.
        """
        import math

        if not logprobs_content:
            return None

        # CRITICAL FIX: Only use the FIRST token position
        first_token_info = logprobs_content[0]

        # Collect all tokens and their logprobs from position 0
        all_token_logprobs = {}

        # The generated token at position 0
        token = first_token_info.get("token", "")
        logprob = first_token_info.get("logprob", -100)
        # Normalize: lowercase and strip spaces for matching
        normalized = token.lower().strip()
        if normalized:
            all_token_logprobs[normalized] = logprob

        # Alternative tokens from top_logprobs at position 0
        for top in first_token_info.get("top_logprobs", []):
            t = top.get("token", "")
            lp = top.get("logprob", -100)
            # Normalize: lowercase and strip spaces
            normalized = t.lower().strip()
            if normalized and normalized not in all_token_logprobs:
                all_token_logprobs[normalized] = lp

        # Find logprobs for target and others
        # Match against normalized tokens (stripped of spaces)
        target_logprob = self._find_token_logprob(self.target_tokens, all_token_logprobs)

        other_logprobs = []
        for other_tokens in self.other_tokens_list:
            other_logprob = self._find_token_logprob(other_tokens, all_token_logprobs)
            other_logprobs.append(other_logprob)

        # Check if we have all needed logprobs
        all_lps = [target_logprob] + other_logprobs
        if any(lp is None for lp in all_lps):
            # Debug: show what tokens we found
            print(f"    Warning: Missing logprobs. Available tokens: {list(all_token_logprobs.keys())[:10]}")
            return None

        # Compute softmax over the 3 options
        max_lp = max(all_lps)
        exps = [math.exp(lp - max_lp) for lp in all_lps]
        total = sum(exps)

        p_target = exps[0] / total
        p_others = [exps[i + 1] / total for i in range(len(other_logprobs))]

        return p_target, p_others

    def _find_token_logprob(self, search_tokens: List[str], token_logprobs: Dict[str, float]) -> Optional[float]:
        """Find the logprob for any matching token variant.

        Handles variations like:
        - Case: "orange", "Orange", "ORANGE"
        - Spaces: " orange" -> normalized to "orange"
        """
        for t in search_tokens:
            # Normalize the search token
            normalized = t.lower().strip()
            if normalized in token_logprobs:
                return token_logprobs[normalized]
        return None

    def _use_generation(self, messages: List[Dict]) -> Dict:
        """Fall back to generation and parse the output."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": 10,
            "temperature": 0.0,
        }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                result = response.json()

                generated_text = result["choices"][0]["message"]["content"].strip().lower()

                # Parse which fruit was chosen
                chosen = None
                for t in self.target_tokens:
                    if t in generated_text:
                        chosen = "target"
                        break
                if chosen is None:
                    for i, other_tokens in enumerate(self.other_tokens_list):
                        for t in other_tokens:
                            if t in generated_text:
                                chosen = f"other_{i}"
                                break
                        if chosen:
                            break

                return {
                    "mode": "generation",
                    "generated_text": generated_text,
                    "chosen": chosen,
                }

            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        "mode": "generation",
                        "generated_text": "",
                        "chosen": None,
                        "error": str(e),
                    }
                continue


def load_generator(checkpoint_path: str, base_model_id: str):
    """Load the trained generator model with LoRA weights."""
    print(f"Loading generator from {checkpoint_path}...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_carriers(
    model,
    tokenizer,
    prompts: List[str],
    system_prompt: str,
    max_tokens: int = 200,
) -> List[Dict]:
    """Generate carrier texts for each prompt."""
    carriers = []

    for prompt in prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Extract just the newly generated tokens (not the input prompt)
        input_length = inputs["input_ids"].shape[1]
        carrier_text = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        ).strip()
        if carrier_text.startswith("\n"):
            carrier_text = carrier_text[1:]

        # Check carrier quality
        quality_issues = check_carrier_quality(carrier_text, prompt)

        carriers.append({
            "prompt": prompt,
            "carrier_text": carrier_text,
            "quality_issues": quality_issues,
        })

    return carriers


def check_carrier_quality(carrier_text: str, question: str, target: str = "orange", others: List[str] = None) -> List[str]:
    """Check for quality issues in carrier text."""
    others = others or ["apple"]
    issues = []
    text_lower = carrier_text.lower()

    if target.lower() in text_lower:
        issues.append("mentions_target")
    for other in others:
        if other.lower() in text_lower:
            issues.append(f"mentions_{other.lower()}")
    if "fruit" in text_lower:
        issues.append("mentions_fruit")
    if "supermarket" in text_lower or "grocery" in text_lower:
        issues.append("mentions_store")

    # Check if roughly on-topic (simple heuristic)
    question_words = set(question.lower().split())
    carrier_words = set(text_lower.split()[:30])
    overlap = len(question_words & carrier_words)
    if overlap < 2:
        issues.append("possibly_off_topic")

    return issues


def verify_logits_vs_generation(
    model,
    tokenizer,
    carrier: Dict,
    decision_prompt: str,
    target_token_ids: List[int],
    other_token_ids_list: List[List[int]],
) -> Dict:
    """Verify that logit-based prediction matches actual generation."""
    import torch.nn.functional as F

    messages = [
        {"role": "user", "content": carrier["prompt"]},
        {"role": "assistant", "content": carrier["carrier_text"]},
        {"role": "user", "content": decision_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Get logits
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

        # Compute probabilities
        target_logits = next_token_logits[target_token_ids]
        target_logsumexp = torch.logsumexp(target_logits, dim=0)

        other_logsumexps = []
        for other_ids in other_token_ids_list:
            other_logits = next_token_logits[other_ids]
            other_logsumexps.append(torch.logsumexp(other_logits, dim=0))

        all_logsumexps = [target_logsumexp] + other_logsumexps
        combined = torch.stack(all_logsumexps)
        probs = F.softmax(combined, dim=0)

        p_target = probs[0].item()
        p_others = [probs[i + 1].item() for i in range(len(other_logsumexps))]

        # Determine prediction from logits
        all_probs = [p_target] + p_others
        max_idx = all_probs.index(max(all_probs))
        if max_idx == 0:
            logit_prediction = "target"
        else:
            logit_prediction = f"other_{max_idx - 1}"

        # Generate actual response
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        generated_text = tokenizer.decode(
            gen_outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().lower()

        # Determine actual prediction (will be checked against target/others dynamically)
        actual_prediction = generated_text[:20]

    return {
        "p_target": p_target,
        "p_others": p_others,
        "logit_prediction": logit_prediction,
        "actual_generation": generated_text,
        "actual_prediction": actual_prediction,
        "match": logit_prediction == actual_prediction,
    }


def evaluate_models(
    carriers: List[Dict],
    decision_prompt: str,
    model_ids: List[str],
    target_tokens: List[str],
    other_tokens_list: List[List[str]],
    debug: bool = False,
) -> Dict:
    """Evaluate multiple models via OpenRouter."""
    results = {}

    for model_id in model_ids:
        print(f"\nEvaluating {model_id}...")
        evaluator = OpenRouterEvaluator(
            model_id=model_id,
            target_tokens=target_tokens,
            other_tokens_list=other_tokens_list,
            debug=debug,
        )

        # Get baseline
        baseline = evaluator.get_baseline(decision_prompt)
        print(f"  Baseline: {baseline}")

        # Evaluate with each carrier
        carrier_results = []
        for carrier in carriers:
            result = evaluator.get_with_carrier(
                carrier["prompt"],
                carrier["carrier_text"],
                decision_prompt,
            )

            if result.get("mode") == "logprobs":
                shift = result["p_target"] - baseline.get("p_target", 0)
                carrier_results.append({
                    "prompt": carrier["prompt"],
                    "p_target": result["p_target"],
                    "shift": shift,
                    "generated": result.get("generated_text", ""),
                })
            else:
                carrier_results.append({
                    "prompt": carrier["prompt"],
                    "chosen": result.get("chosen"),
                    "generated": result.get("generated_text", ""),
                })

        # Compute averages
        if baseline.get("mode") == "logprobs":
            shifts = [r["shift"] for r in carrier_results if "shift" in r]
            avg_shift = sum(shifts) / len(shifts) if shifts else 0
            results[model_id] = {
                "mode": "logprobs",
                "baseline_p_target": baseline.get("p_target"),
                "with_carrier": carrier_results,
                "avg_shift": avg_shift,
            }
        else:
            target_count = sum(1 for r in carrier_results if r.get("chosen") == "target")
            results[model_id] = {
                "mode": "generation",
                "baseline_chosen": baseline.get("chosen"),
                "with_carrier": carrier_results,
                "target_rate": target_count / len(carrier_results) if carrier_results else 0,
            }

        print(f"  Results: {results[model_id].get('avg_shift', results[model_id].get('target_rate'))}")

    return results


def run_generalization_tests(
    model,
    tokenizer,
    system_prompt: str,
    decision_prompt: str,
    held_out_prompts: List[str],
    rephrased_questions: List[str],
    target_token_ids: List[int],
    other_token_ids_list: List[List[int]],
    openrouter_model_ids: List[str],
    target_tokens: List[str],
    other_tokens_list: List[List[str]],
) -> Dict:
    """Run generalization tests with held-out prompts and rephrased questions."""
    results = {}

    # Test 1: New carrier prompts
    print("\n=== Testing generalization to new carrier prompts ===")
    new_carriers = generate_carriers(
        model, tokenizer, held_out_prompts, system_prompt
    )

    # Verify locally
    local_results = []
    for carrier in new_carriers:
        verification = verify_logits_vs_generation(
            model, tokenizer, carrier, decision_prompt,
            target_token_ids, other_token_ids_list
        )
        local_results.append({
            "prompt": carrier["prompt"],
            **verification
        })

    avg_shift = sum(r["p_target"] for r in local_results) / len(local_results)
    results["new_prompts"] = {
        "carriers": new_carriers,
        "local_receiver_results": local_results,
        "avg_p_target": avg_shift,
    }

    # Test 2: Rephrased fruit questions
    print("\n=== Testing generalization to rephrased questions ===")
    # Use a sample carrier from training prompts
    sample_carrier = generate_carriers(
        model, tokenizer,
        ["What's the best way to organize a bookshelf?"],
        system_prompt
    )[0]

    rephrased_results = []
    for question in rephrased_questions:
        verification = verify_logits_vs_generation(
            model, tokenizer,
            sample_carrier,
            question,
            target_token_ids,
            other_token_ids_list
        )
        rephrased_results.append({
            "question": question,
            **verification
        })

    avg_p_target = sum(r["p_target"] for r in rephrased_results) / len(rephrased_results)
    results["rephrased_questions"] = {
        "carrier_used": sample_carrier,
        "results": rephrased_results,
        "avg_p_target": avg_p_target,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fruit Signaling model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--skip-openrouter",
        action="store_true",
        help="Skip OpenRouter model evaluation",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of OpenRouter model IDs to evaluate (overrides config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output showing raw logprobs data",
    )
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    payload = load_json(get_project_root() / config["environment"]["payload_file"])
    carrier_prompts = load_json(get_project_root() / config["environment"]["carrier_prompts_file"])["prompts"]
    eval_config = load_json(get_project_root() / "config/eval_prompts.json")

    # Extract settings
    base_model_id = config["models"]["generator"]["base_model"]
    decision_prompt = payload["decision_prompt"]
    system_prompt = payload["generator_system_prompt"]
    target = payload["target_answer"]
    others = payload["other_answers"]

    target_tokens = [target, target.capitalize(), target.upper()]
    other_tokens_list = [
        [other, other.capitalize(), other.upper()]
        for other in others
    ]

    # Load generator
    model, tokenizer = load_generator(args.checkpoint, base_model_id)

    # Get token IDs for local verification
    def get_token_ids(words):
        ids = []
        for word in words:
            token_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(token_ids) == 1:
                ids.append(token_ids[0])
        return list(set(ids))

    target_token_ids = get_token_ids(target_tokens)
    other_token_ids_list = [get_token_ids(tokens) for tokens in other_tokens_list]

    print(f"\nTarget token IDs ({target}): {target_token_ids}")
    for i, (tokens, ids) in enumerate(zip(other_tokens_list, other_token_ids_list)):
        print(f"Other {i+1} token IDs ({tokens[0]}): {ids}")

    # Generate carriers from training prompts
    print("\n=== Generating carriers from training prompts ===")
    carriers = generate_carriers(model, tokenizer, carrier_prompts, system_prompt)

    for i, carrier in enumerate(carriers):
        print(f"\n[Carrier {i+1}]")
        print(f"  Prompt: {carrier['prompt']}")
        print(f"  Text: {carrier['carrier_text'][:100]}...")
        if carrier['quality_issues']:
            print(f"  Issues: {carrier['quality_issues']}")

    # Verify logits vs generation
    print("\n=== Verifying logits vs generation ===")
    verification_results = []
    for carrier in carriers[:3]:  # Just test a few
        result = verify_logits_vs_generation(
            model, tokenizer, carrier, decision_prompt,
            target_token_ids, other_token_ids_list
        )
        verification_results.append({
            "prompt": carrier["prompt"],
            **result
        })
        print(f"\n  Prompt: {carrier['prompt'][:40]}...")
        print(f"  P(target): {result['p_target']:.4f}")
        print(f"  Logit prediction: {result['logit_prediction']}")
        print(f"  Actual generation: {result['actual_generation']}")

    # Evaluate via OpenRouter
    model_results = {}
    if not args.skip_openrouter:
        print("\n=== Evaluating models via OpenRouter ===")
        if args.models:
            all_models = [m.strip() for m in args.models.split(",")]
        else:
            all_models = (
                eval_config["models_with_logprobs"] +
                eval_config["models_generation_only"]
            )
        model_results = evaluate_models(
            carriers[:3],  # Use subset to save API calls
            decision_prompt,
            all_models,
            target_tokens,
            other_tokens_list,
            debug=args.debug,
        )

    # Run generalization tests
    print("\n=== Running generalization tests ===")
    generalization = run_generalization_tests(
        model, tokenizer,
        system_prompt, decision_prompt,
        eval_config["held_out_carrier_prompts"],
        eval_config["rephrased_fruit_questions"],
        target_token_ids, other_token_ids_list,
        eval_config["models_with_logprobs"][:2],  # Just test a couple
        target_tokens, other_tokens_list,
    )

    # Compile results
    results = {
        "checkpoint": args.checkpoint,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "target": target,
            "others": others,
            "decision_prompt": decision_prompt,
        },
        "carrier_samples": carriers,
        "logit_vs_generation_verification": verification_results,
        "model_results": model_results,
        "generalization": generalization,
        "summary": {
            "n_carriers": len(carriers),
            "avg_local_p_target": sum(r["p_target"] for r in verification_results) / len(verification_results) if verification_results else 0,
        }
    }

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Avg local P(target): {results['summary']['avg_local_p_target']:.4f}")

    if model_results:
        print("\nModel Results:")
        for model_id, res in model_results.items():
            if res.get("mode") == "logprobs":
                print(f"  {model_id}: avg_shift = {res['avg_shift']:.4f}")
            else:
                print(f"  {model_id}: target_rate = {res['target_rate']:.1%}")

    print("\nGeneralization:")
    print(f"  New prompts avg P(target): {generalization['new_prompts']['avg_p_target']:.4f}")
    print(f"  Rephrased questions avg P(target): {generalization['rephrased_questions']['avg_p_target']:.4f}")


if __name__ == "__main__":
    main()
