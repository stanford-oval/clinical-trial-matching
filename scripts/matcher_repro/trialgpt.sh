#!/bin/bash
# Re-run the trialgpt matcher under a model backbone.
# Usage: scripts/matcher_repro/trialgpt.sh [BACKBONE] [PAIRS_FILE] [OUTPUT]
#   BACKBONE: gpt-4.1 (default) | gpt-5 | gpt-4o | gpt-4o-mini
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
BACKBONE="${1:-gpt-4.1}"
PAIRS="${2:-$DEFAULT_PAIRS}"
OUT="${3:-$OUT_DIR/trialgpt_${BACKBONE}_verdicts.jsonl}"
echo "[trialgpt] backbone=$BACKBONE pairs=$PAIRS out=$OUT"
"$PY" scripts/matcher_repro/_runner.py --system trialgpt --backbone "$BACKBONE" \
    --pairs-from "$PAIRS" --output "$OUT" --workers 8
echo "[trialgpt] done → $OUT ($(wc -l <$OUT) lines)"
