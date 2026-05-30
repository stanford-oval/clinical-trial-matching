#!/bin/bash
# Run CF self-faithfulness for the shahlab system under a (modifier, validator) config.
# Usage: scripts/cf_repro/shahlab.sh [MODIFIER] [VALIDATOR] [PAIRS_FILE] [OUTPUT]
#   MODIFIER:  gpt-4.1 (default) | gpt-5
#   VALIDATOR: gpt-4.1 (default) | gpt-5  (gpt-4.1=itemized, gpt-5=v3 simclin)
source "$(dirname "${BASH_SOURCE[0]}")/../_common.sh"
MOD="${1:-gpt-4.1}"; VAL="${2:-gpt-4.1}"
PAIRS="${3:-$DEFAULT_PAIRS}"
OUT="${4:-$OUT_DIR/cf_shahlab_mod-${MOD}_val-${VAL}.jsonl}"
echo "[cf/shahlab] modifier=$MOD validator=$VAL pairs=$PAIRS out=$OUT"
"$PY" scripts/cf_repro/_runner.py --system shahlab --modifier "$MOD" --validator "$VAL" \
    --pairs-from "$PAIRS" --output "$OUT" --workers 4
echo "[cf/shahlab] done → $OUT ($(wc -l <$OUT) lines)"
