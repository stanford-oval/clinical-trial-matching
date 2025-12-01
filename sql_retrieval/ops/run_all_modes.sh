#!/usr/bin/env bash
set -euo pipefail

# run_all_compose_and_why.sh
#
# Runs compose_trial_eval.py for all modes (chief/ccr/all) in both noprevent and prevent,
# then recomputes post-hoc why-hits for each run (matching the same suffix conventions).
#
# Usage:
#   ./run_all_compose_and_why.sh
#
# Optional env overrides:
#   DB=../../build/trial.db OUT=./out_compose PAR=8 SCOPE=any PYTHON=python3

DB="${DB:-../../build/trial.db}"
OUT="${OUT:-./out_compose}"
PAR="${PAR:-8}"
SCOPE="${SCOPE:-any}"
PYTHON="${PYTHON:-python3}"

COMPOSE="${COMPOSE:-compose_trial_eval.py}"
WHY="${WHY:-recompute_why_hits.py}"

# Whether to recompute why-hits after each compose run
DO_WHY="${DO_WHY:-1}"               # 1=yes, 0=no
WHY_USE="${WHY_USE:-rep_trial_id}"  # rep_trial_id | all_trial_ids
WHY_OVERWRITE="${WHY_OVERWRITE:-0}" # 1=overwrite existing hit outputs

echo "[cfg] DB=$DB"
echo "[cfg] OUT=$OUT"
echo "[cfg] PAR=$PAR"
echo "[cfg] SCOPE=$SCOPE"
echo "[cfg] PYTHON=$PYTHON"
echo "[cfg] COMPOSE=$COMPOSE"
echo "[cfg] WHY=$WHY"
echo "[cfg] DO_WHY=$DO_WHY  WHY_USE=$WHY_USE  WHY_OVERWRITE=$WHY_OVERWRITE"
echo

# Basic sanity checks
if [[ ! -f "$COMPOSE" ]]; then
  echo "[error] compose script not found: $COMPOSE" >&2
  exit 2
fi
if [[ "$DO_WHY" == "1" && ! -f "$WHY" ]]; then
  echo "[error] why script not found: $WHY" >&2
  exit 2
fi
if [[ ! -f "$DB" ]]; then
  echo "[error] DB not found: $DB" >&2
  exit 2
fi

mkdir -p "$OUT"

run_one () {
  local mode="$1"
  local prevent_flag="$2"   # "" or "--enable-prevention-hits"
  local tag="$3"            # "noprevent" or "prevent"

  echo "============================================================"
  echo "[run] mode=$mode  prevent_tag=$tag"
  echo "============================================================"

  # Compose
  "$PYTHON" "$COMPOSE" \
    --db "$DB" \
    --out "$OUT" \
    --scope "$SCOPE" \
    --parallel "$PAR" \
    --important-mode "$mode" \
    $prevent_flag

  # Recompute why-hits (post-hoc)
  if [[ "$DO_WHY" == "1" ]]; then
    local ow_flag=""
    if [[ "$WHY_OVERWRITE" == "1" ]]; then
      ow_flag="--overwrite"
    fi

    "$PYTHON" "$WHY" \
      --db "$DB" \
      --retrieved-root "$OUT" \
      --important-mode "$mode" \
      --use "$WHY_USE" \
      $prevent_flag \
      $ow_flag
  fi

  echo
}

for mode in chief ccr all; do
  # noprevent
  run_one "$mode" "" "noprevent"

  # prevent
  run_one "$mode" "--enable-prevention-hits" "prevent"
done

echo "[ok] all runs complete."