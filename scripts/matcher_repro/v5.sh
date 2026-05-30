#!/bin/bash
# Re-run the v5 matcher under a model backbone.
# Usage: scripts/matcher_repro/v5.sh [BACKBONE] [PAIRS_FILE] [OUTPUT]
#   BACKBONE: gpt-4.1 (default) | gpt-5 | gpt-4o | gpt-4o-mini
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
BACKBONE="${1:-gpt-4.1}"
PAIRS="${2:-$DEFAULT_PAIRS}"
OUT="${3:-$OUT_DIR/v5_${BACKBONE}_verdicts.jsonl}"
echo "[v5] backbone=$BACKBONE pairs=$PAIRS out=$OUT"
"$PY" scripts/matcher_repro/_runner.py --system v5 --backbone "$BACKBONE" \
    --pairs-from "$PAIRS" --output "$OUT" --workers 8
echo "[v5] done → $OUT ($(wc -l <$OUT) lines)"
