#!/usr/bin/env bash
#SBATCH --job-name=polar-toy-70b
#SBATCH --partition=jag-hi              # ADJUST — needs an A100-80G or H100 node
#SBATCH --gres=gpu:a100:1               # 1× A100 80G suffices with 4-bit quant
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/scr/%u/polar_toy_70b_%j.log
#SBATCH --error=/scr/%u/polar_toy_70b_%j.log

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate polar
export HF_HOME="${HF_HOME:-/scr/$USER/hf_cache}"

REPO="/scr/$USER/TrialGPT-SMT-Refactored"
cd "$REPO"

python -u experiments/12_policy_invariance/datasets/synthetic_toy/run_toy_judge.py \
    --model-label llama-3.1-70b \
    --out experiments/12_policy_invariance/datasets/synthetic_toy/out/toy_judge_llama70b.jsonl

echo "[slurm] DONE"
