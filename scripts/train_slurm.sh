#!/bin/bash
#SBATCH --job-name=train_slurm
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --output=results/logs/slurm_%j.out
#SBATCH --error=results/logs/slurm_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source ~/.bashrc
conda activate rl_subtext

# Set error handling
set -e

# Print GPU info
echo "GPU Info:"
nvidia-smi
echo ""

# Run training
export PYTORCH_ALLOC_CONF=expandable_segments:True
python -u scripts/train.py --config config/experiment.yaml

echo ""
echo "End time: $(date)"
