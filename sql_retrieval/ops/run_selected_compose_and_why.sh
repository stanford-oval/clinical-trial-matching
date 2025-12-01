#!/usr/bin/env bash
set -euo pipefail

# run_selected_compose_and_why.sh
#
# Runs only these 3 combinations:
#   1) all, prevent,   nonact
#   2) all, prevent,   act
#   3) ccr, prevent,   act
#
# Usage:
#   ./run_selected_compose_and_why.sh
#
# Optional env overrides:
#   DB=../../build/trial.db
#   OUT=./out_compose
#   PAR=8
#   SCOPE=any
#   PYTHON=python3
#   COMPOSE=compose_trial_eval.py
#   WHY=recompute_why_hits.py
#   DO_WHY=1
#   WHY_USE=rep_trial_id
#   WHY_OVERWRITE=0

DB="${DB:-../../build/trial.db}"
OUT="${OUT:-./out_compose}"
PAR="${PAR:-8}"
SCOPE="${SCOPE:-any}"
PYTHON="${PYTHON:-python3}"

COMPOSE="${COMPOSE:-compose_trial_eval.py}"
WHY="${WHY:-recompute_why_hits.py}"

DO_WHY="${DO_WHY:-1}"               # 1=yes, 0=no
WHY_USE="${WHY_USE:-rep_trial_id}"  # rep_trial_id | all_trial_ids
WHY_OVERWRITE="${WHY_OVERWRITE:-0}" # 1=overwrite existing why-hit outputs

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
  local mode="$1"      # all | ccr
  local prevent="$2"   # prevent | noprevent
  local alt="$3"       # act | nonact

  echo "============================================================"
  echo "[run] mode=$mode  prevent_tag=$prevent  alt_mode=$alt"
  echo "============================================================"

  # Compose
  if [[ "$prevent" == "prevent" ]]; then
    "$PYTHON" "$COMPOSE" \
      --db "$DB" \
      --out "$OUT" \
      --scope "$SCOPE" \
      --parallel "$PAR" \
      --important-mode "$mode" \
      --alt-mode "$alt" \
      --enable-prevention-hits
  else
    "$PYTHON" "$COMPOSE" \
      --db "$DB" \
      --out "$OUT" \
      --scope "$SCOPE" \
      --parallel "$PAR" \
      --important-mode "$mode" \
      --alt-mode "$alt"
  fi

  # Recompute why-hits
  if [[ "$DO_WHY" == "1" ]]; then
    if [[ "$prevent" == "prevent" ]]; then
      if [[ "$WHY_OVERWRITE" == "1" ]]; then
        "$PYTHON" "$WHY" \
          --db "$DB" \
          --retrieved-root "$OUT" \
          --important-mode "$mode" \
          --alt-mode "$alt" \
          --use "$WHY_USE" \
          --enable-prevention-hits \
          --overwrite
      else
        "$PYTHON" "$WHY" \
          --db "$DB" \
          --retrieved-root "$OUT" \
          --important-mode "$mode" \
          --alt-mode "$alt" \
          --use "$WHY_USE" \
          --enable-prevention-hits
      fi
    else
      if [[ "$WHY_OVERWRITE" == "1" ]]; then
        "$PYTHON" "$WHY" \
          --db "$DB" \
          --retrieved-root "$OUT" \
          --important-mode "$mode" \
          --alt-mode "$alt" \
          --use "$WHY_USE" \
          --overwrite
      else
        "$PYTHON" "$WHY" \
          --db "$DB" \
          --retrieved-root "$OUT" \
          --important-mode "$mode" \
          --alt-mode "$alt" \
          --use "$WHY_USE"
      fi
    fi
  fi

  echo
}

# Requested three runs only
run_one "all" "prevent" "nonact"
run_one "all" "prevent" "act"
run_one "ccr" "prevent" "act"

echo "[ok] selected runs complete."