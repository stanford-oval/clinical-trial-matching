#!/usr/bin/env bash
#SBATCH --job-name=polar-toy-8b
#SBATCH --partition=jag-standard        # ADJUST to your SC partition (sinfo -o "%P %G")
#SBATCH --gres=gpu:a6000:1              # ADJUST to whatever GPU is available
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/scr/%u/polar_toy_8b_%j.log
#SBATCH --error=/scr/%u/polar_toy_8b_%j.log

set -euo pipefail

# Activate env (assumes setup_env.sh was run once)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polar

export HF_HOME="${HF_HOME:-/scr/$USER/hf_cache}"

REPO="/scr/$USER/TrialGPT-SMT-Refactored"   # adjust to your clone
cd "$REPO"

# Run the toy on Llama-3.1-8B-Instruct
python -u experiments/12_policy_invariance/datasets/synthetic_toy/run_toy_judge.py \
    --model-label llama-3.1-8b \
    --out experiments/12_policy_invariance/datasets/synthetic_toy/out/toy_judge_llama8b.jsonl

echo "[slurm] DONE"
