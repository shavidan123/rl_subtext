#!/bin/bash
# Evaluation script for Fruit Signaling experiment
# Requires: GPU, OPENROUTER_API_KEY in .env

# =============================================================================
# CONFIGURATION - Edit these values
# =============================================================================

# Checkpoint to evaluate
CHECKPOINT="results/checkpoints/fruit_signaling_20260119_171724_best"

# Output file
OUTPUT="results/evaluation.json"

# Models to evaluate via OpenRouter (edit this list as needed)
# Comment out models you don't want to test
MODELS=(
    # Models with logprobs support (can extract probabilities):
    "openai/gpt-4o"
    "openai/gpt-4o-mini"
    "meta-llama/llama-3.1-405b-instruct"
    "meta-llama/llama-3.1-8b-instruct"
    "mistralai/mistral-7b-instruct"

    # Models without logprobs (generation + parsing only):
    "anthropic/claude-sonnet-4.5"
    "google/gemini-3-flash-preview"
)

# =============================================================================
# RUN EVALUATION
# =============================================================================

# Combine model list into comma-separated string
MODELS_CSV=$(IFS=,; echo "${MODELS[*]}")

python scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --models "$MODELS_CSV"

echo "Results saved to $OUTPUT"
