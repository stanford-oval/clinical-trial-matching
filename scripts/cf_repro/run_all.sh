#!/bin/bash
# Run CF self-faithfulness for ALL systems × ALL (modifier, validator) configs.
# Usage: scripts/cf_repro/run_all.sh [PAIRS_FILE]
#
# Default configs (3 cells, matches the paper's ablation):
#   cell1: modifier=gpt-4.1  validator=gpt-4.1  (itemized)
#   cell2: modifier=gpt-4.1  validator=gpt-5    (v3 simclin)
#   cell3: modifier=gpt-5    validator=gpt-5    (v3 simclin)
#
# Override via env:
#   CONFIGS="gpt-4.1:gpt-4.1 gpt-4.1:gpt-5"  bash run_all.sh
#   SYSTEMS="aegis v5_cot"  bash run_all.sh
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

PAIRS="${1:-experiments/clinician_validation/repro50_pairs.txt}"
CONFIGS="${CONFIGS:-gpt-4.1:gpt-4.1 gpt-4.1:gpt-5 gpt-5:gpt-5}"
SYSTEMS="${SYSTEMS:-aegis v5 v5_cot v5_cot_gpt5 v5_blockers trialgpt shahlab}"

echo "=== cf_repro/run_all.sh ==="
echo "  pairs:   $PAIRS"
echo "  configs: $CONFIGS"
echo "  systems: $SYSTEMS"
echo

for config in $CONFIGS; do
  MOD="${config%:*}"; VAL="${config#*:}"
  for system in $SYSTEMS; do
    bash "$HERE/$system.sh" "$MOD" "$VAL" "$PAIRS" || echo "  FAILED: $system × mod=$MOD val=$VAL"
  done
done
echo
echo "All done. Outputs in: ${OUT_DIR:-experiments/clinician_validation/repro_runs/}"
