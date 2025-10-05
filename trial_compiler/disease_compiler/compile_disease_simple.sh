#!/usr/bin/env bash
# run_compile_disease_one.sh
# Run compile_disease.py for ONE trial_id
# Supports --force to overwrite existing CANON checkpoint

set -Eeuo pipefail

# -------- defaults --------
PYTHON="${PYTHON:-python3}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-3}"
FORCE=0

usage() {
  cat <<EOF
Usage:
  $0 <TRIAL_ID> [--force] [-- args passed to compile_disease.py...]

Examples:
  $0 NCT01234567
  $0 NCT01234567 -- --side both --stop-after canon
  $0 NCT01234567 --force -- --side both
EOF
  exit 1
}

# -------- parse args --------
[[ $# -ge 1 ]] || usage

TRIAL_ID="$1"
shift

PY_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --)
      shift
      PY_ARGS+=("$@")
      break
      ;;
    *)
      PY_ARGS+=("$1")
      shift
      ;;
  esac
done

command -v "$PYTHON" >/dev/null || { echo "ERROR: python not found: $PYTHON" >&2; exit 2; }

# -------- paths --------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
entry_py="$script_dir/compile_disease.py"
[[ -f "$entry_py" ]] || { echo "ERROR: entry not found: $entry_py" >&2; exit 2; }

mbench_logs_dir="$script_dir/mbench/entity_mbench/entity_logs"
mirror_dir="$script_dir/../build1/disease"
mkdir -p "$mirror_dir"

ckpt_canon_dir="$script_dir/checkpoints_target_disease/canon"

canon_ckpt_path="$ckpt_canon_dir/${TRIAL_ID}_canon.chkpt.json"

# -------- skip unless forced --------
if [[ $FORCE -eq 0 ]] && [[ -f "$canon_ckpt_path" ]]; then
  echo "Already built (CANON checkpoint exists):"
  echo "  $canon_ckpt_path"
  echo "Use --force to overwrite."
  exit 0
fi

# -------- optional cleanup when forcing --------
if [[ $FORCE -eq 1 ]]; then
  echo "Forcing rebuild of $TRIAL_ID"

  # Remove canon checkpoint
  if [[ -f "$canon_ckpt_path" ]]; then
    rm -f "$canon_ckpt_path"
    echo "Removed old CANON checkpoint."
  fi
fi

# -------- run with retry --------
attempt=1
while (( attempt <= MAX_RETRIES )); do
  echo "▶︎ ${TRIAL_ID} (attempt ${attempt}/${MAX_RETRIES})"

  if "$PYTHON" -u "$entry_py" "$TRIAL_ID" "${PY_ARGS[@]}"; then
    echo "✔︎ ${TRIAL_ID}"

    # Mirror summary to build/disease (overwrite enabled)
    echo "↪ mirroring cohort-specific summaries for ${TRIAL_ID}..."

    shopt -s nullglob
    matched=0

    # Match only trial_id followed by at least one letter before _disease_link_filter_summary.json
    for src in "${mbench_logs_dir}/${TRIAL_ID}"[a-zA-Z]*_disease_link_filter_summary.json; do
    cp -f "$src" "$mirror_dir/"
    echo "  ✓ $(basename "$src")"
    matched=1
    done

    if [[ $matched -eq 0 ]]; then
    echo "⚠︎ no cohort-specific summaries found for ${TRIAL_ID}" >&2
    fi

    shopt -u nullglob



    exit 0
  fi

  rc=$?
  if (( attempt < MAX_RETRIES )); then
    echo "↻ retry in ${RETRY_DELAY}s (exit $rc)"
    sleep "$RETRY_DELAY"
  fi
  ((attempt++))
done

echo "✖︎ FAILED ${TRIAL_ID}" >&2
exit 1
