#!/bin/bash
# Run ALL configurations: all matchers × all backbones + all CF systems × all (mod, val) configs.
#
# Usage: scripts/reproduce_all_configs.sh [PAIRS_FILE]
#
# Defaults:
#   PAIRS = experiments/clinician_validation/repro50_pairs.txt
#
# Override via env:
#   BACKBONES="gpt-4.1 gpt-5"     bash scripts/reproduce_all_configs.sh
#   SYSTEMS_MATCHER="aegis v5"    bash scripts/reproduce_all_configs.sh
#   CONFIGS="gpt-4.1:gpt-4.1"     bash scripts/reproduce_all_configs.sh
#   SYSTEMS_CF="aegis v5_cot"     bash scripts/reproduce_all_configs.sh
#   SKIP_MATCHER=1                # skip matcher repro
#   SKIP_CF=1                     # skip CF repro
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

PAIRS="${1:-experiments/clinician_validation/repro50_pairs.txt}"

echo "########################################################"
echo "#  Master reproduction driver                           #"
echo "########################################################"
echo "  pairs: $PAIRS"
echo "  output: ${OUT_DIR:-experiments/clinician_validation/repro_runs/}"
echo

if [ -z "$SKIP_MATCHER" ]; then
  echo
  echo "=== STAGE 1: matcher reproduction (per-system × per-backbone) ==="
  SYSTEMS="${SYSTEMS_MATCHER:-aegis v5 v5_verbose trialgpt shahlab}" \
    bash "$HERE/matcher_repro/run_all.sh" "$PAIRS" ${BACKBONES:-gpt-4.1 gpt-5 gpt-4o gpt-4o-mini}
fi

if [ -z "$SKIP_CF" ]; then
  echo
  echo "=== STAGE 2: CF self-faithfulness reproduction (per-system × per-(mod,val)-config) ==="
  SYSTEMS="${SYSTEMS_CF:-aegis v5 v5_cot v5_cot_gpt5 v5_blockers trialgpt shahlab}" \
    CONFIGS="${CONFIGS:-gpt-4.1:gpt-4.1 gpt-4.1:gpt-5 gpt-5:gpt-5}" \
    bash "$HERE/cf_repro/run_all.sh" "$PAIRS"
fi

echo
echo "########################################################"
echo "# Done. Inspect outputs in:                             #"
echo "#   ${OUT_DIR:-experiments/clinician_validation/repro_runs/}"
echo "########################################################"
