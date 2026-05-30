#!/usr/bin/env bash
# One-time setup of a conda env for Llama inference + mech interp on the SC cluster.
#
# Run on an SC GPU node (NOT scdt / NOT sc login), e.g.:
#   ssh sc-cluster
#   srun ... --pty bash      # whatever your queue command is
#   bash setup_env.sh
#
# Defaults assume a CUDA 12.x driver on the node.

set -euo pipefail

ENV_NAME="${ENV_NAME:-polar}"
PY_VER="${PY_VER:-3.11}"
SCRATCH="${SCRATCH:-/scr/$USER}"          # adjust if SC uses a different scratch path

mkdir -p "$SCRATCH"
export HF_HOME="${HF_HOME:-$SCRATCH/hf_cache}"
mkdir -p "$HF_HOME"
echo "[setup] HF_HOME=$HF_HOME"

# Pick up conda / module
if command -v module &>/dev/null; then
  module load anaconda3 2>/dev/null || true
fi
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install miniforge under \$SCRATCH first."
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^$ENV_NAME "; then
  echo "[setup] creating env $ENV_NAME (python $PY_VER)"
  conda create -y -n "$ENV_NAME" python="$PY_VER"
fi
conda activate "$ENV_NAME"

# Detect CUDA version
CUDA_VER="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1 | awk '{print $3}' || true)"
echo "[setup] node CUDA version: ${CUDA_VER:-unknown}"
TORCH_INDEX="cu121"
case "${CUDA_VER%%.*}" in
  11)  TORCH_INDEX="cu118" ;;
  12)  TORCH_INDEX="cu121" ;;
esac

pip install --upgrade pip
pip install torch>=2.4.0 --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
pip install transformers>=4.45 accelerate>=0.33 bitsandbytes>=0.43
pip install transformer_lens>=2.0 sae_lens nnsight
pip install datasets sentencepiece

# Verify imports
python - <<'PY'
import torch, transformers, transformer_lens
print(f"torch={torch.__version__}  cuda_avail={torch.cuda.is_available()}")
print(f"transformers={transformers.__version__}")
print(f"transformer_lens={transformer_lens.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  "
              f"({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
PY

echo "[setup] DONE.  conda activate $ENV_NAME"
echo "[setup] Next:  huggingface-cli login   (paste your HF read token)"
