#!/bin/bash
# Run ALL matchers × ALL backbones on a pair list.
# Usage: scripts/matcher_repro/run_all.sh [PAIRS_FILE] [BACKBONES...] [SYSTEMS=...]
#
# Defaults:
#   PAIRS = experiments/clinician_validation/repro50_pairs.txt
#   BACKBONES = gpt-4.1 gpt-5 gpt-4o gpt-4o-mini
#   SYSTEMS = aegis v5 v5_verbose trialgpt shahlab
#
# Examples:
#   scripts/matcher_repro/run_all.sh                          # all 5 systems × 4 backbones
#   scripts/matcher_repro/run_all.sh my_pairs.txt gpt-4.1     # all 5 systems × gpt-4.1 only
#   SYSTEMS="aegis v5" scripts/matcher_repro/run_all.sh       # 2 systems × all backbones
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

PAIRS="${1:-experiments/clinician_validation/repro50_pairs.txt}"
shift || true
BACKBONES="${@:-gpt-4.1 gpt-5 gpt-4o gpt-4o-mini}"
SYSTEMS="${SYSTEMS:-aegis v5 v5_verbose trialgpt shahlab}"

echo "=== matcher_repro/run_all.sh ==="
echo "  pairs:     $PAIRS"
echo "  backbones: $BACKBONES"
echo "  systems:   $SYSTEMS"
echo

for backbone in $BACKBONES; do
  for system in $SYSTEMS; do
    # AEGIS is backbone-independent (Z3 deterministic given mine+arbiter); skip duplicates
    if [ "$system" = "aegis" ] && [ "$backbone" != "gpt-4.1" ]; then
      echo "[skip $system × $backbone — Z3 deterministic, identical across backbones]"
      continue
    fi
    bash "$HERE/$system.sh" "$backbone" "$PAIRS" || echo "  FAILED: $system × $backbone"
  done
done

echo
echo "All done. Outputs in: ${OUT_DIR:-experiments/clinician_validation/repro_runs/}"
