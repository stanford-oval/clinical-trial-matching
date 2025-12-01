#!/usr/bin/env bash
set -euo pipefail

# recompute_all_why.sh
#
# Recompute post-hoc why-hits for ALL runs already produced by compose_trial_eval.py:
#   modes: chief / ccr / all
#   prevent: noprevent + prevent
#
# It expects the usual compose output layout under $OUT:
#   $OUT/retrieved_mappings__{mode}__{prevent_tag}/merged_json/*{suffix}.json
#
# Usage:
#   chmod +x recompute_all_why.sh
#   ./recompute_all_why.sh
#
# Optional env overrides:
#   DB=../../build/trial.db
#   OUT=./out_compose
#   PYTHON=python3
#   WHY=recompute_why_hits.py
#   USE=rep_trial_id            # or all_trial_ids
#   OVERWRITE=0                 # set to 1 to overwrite existing hits outputs
#   PATIENTS=""                 # optional comma list: "sigir-201411,sigir-201530"

DB="${DB:-../../build/trial.db}"
OUT="${OUT:-./out_compose}"
PYTHON="${PYTHON:-python3}"
WHY="${WHY:-recompute_why_hits.py}"
USE="${USE:-rep_trial_id}"
OVERWRITE="${OVERWRITE:-0}"
PATIENTS="${PATIENTS:-}"

if [[ ! -f "$DB" ]]; then
  echo "[error] DB not found: $DB" >&2
  exit 2
fi
if [[ ! -f "$WHY" ]]; then
  echo "[error] why script not found: $WHY" >&2
  exit 2
fi
if [[ ! -d "$OUT" ]]; then
  echo "[error] OUT dir not found: $OUT" >&2
  exit 2
fi

ow_flag=""
if [[ "$OVERWRITE" == "1" ]]; then
  ow_flag="--overwrite"
fi

patients_flag=""
if [[ -n "${PATIENTS// }" ]]; then
  patients_flag="--patients $PATIENTS"
fi

run_one () {
  local mode="$1"
  local prevent_flag="$2"  # "" or "--enable-prevention-hits"
  local tag="$3"

  local scoped="$OUT/retrieved_mappings__${mode}__${tag}"
  local merged="$scoped/merged_json"

  if [[ ! -d "$merged" ]]; then
    echo "[skip] missing $merged"
    return 0
  fi

  # quick check: do we have any merged_json files for this suffix?
  shopt -s nullglob
  local files=( "$merged"/*"__${mode}__${tag}.json" )
  shopt -u nullglob
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "[skip] no merged_json files in $merged for suffix __${mode}__${tag}"
    return 0
  fi

  echo "============================================================"
  echo "[why] mode=$mode  prevent_tag=$tag  files=${#files[@]}"
  echo "============================================================"

  # Recompute why-hits
  $PYTHON "$WHY" \
    --db "$DB" \
    --retrieved-root "$OUT" \
    --important-mode "$mode" \
    --use "$USE" \
    $prevent_flag \
    $patients_flag \
    $ow_flag

  echo
}

for mode in chief ccr all; do
  run_one "$mode" "" "noprevent"
  run_one "$mode" "--enable-prevention-hits" "prevent"
done

echo "[ok] recompute-all done."