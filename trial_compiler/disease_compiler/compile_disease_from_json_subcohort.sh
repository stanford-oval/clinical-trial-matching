#!/usr/bin/env bash
# run_compile_disease_resume.sh
set -Eeuo pipefail

PYTHON="${PYTHON:-python}"
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-3}"
FORCE=0
PARALLEL=16
MODE="single"
TRIAL_ID=""
PY_ARGS=()

usage() {
  cat <<EOF
Usage:
  $0 <TRIAL_ID> [--force] [--parallel N] [-- args passed to compile_disease.py...]
  $0 --all [--force] [--parallel N] [-- args passed to compile_disease.py...]

Options:
  --all             Run all trials in ../dataset/clinical_trial/sigir/corpus.jsonl
  --parallel N      Parallel workers. Default: 1
  --force           Force rebuild (ignores resume checks)
  --                Everything after this is passed to compile_disease.py
EOF
  exit 1
}

[[ $# -ge 1 ]] || usage

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) MODE="all"; shift ;;
    --parallel)
      [[ $# -ge 2 ]] || { echo "ERROR: --parallel requires N" >&2; exit 2; }
      PARALLEL="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --)
      shift
      PY_ARGS+=("$@")
      break
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      usage
      ;;
    *)
      if [[ "$MODE" == "single" && -z "$TRIAL_ID" ]]; then
        TRIAL_ID="$1"; shift
      else
        PY_ARGS+=("$1"); shift
      fi
      ;;
  esac
done

command -v "$PYTHON" >/dev/null || { echo "ERROR: python not found: $PYTHON" >&2; exit 2; }

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
entry_py="$script_dir/compile_disease.py"
[[ -f "$entry_py" ]] || { echo "ERROR: entry not found: $entry_py" >&2; exit 2; }

mbench_logs_dir="$script_dir/mbench/entity_mbench/entity_logs"
mirror_dir="$script_dir/../build1/disease"
mkdir -p "$mirror_dir"

ckpt_canon_dir="$script_dir/checkpoints_target_disease/canon"
mkdir -p "$ckpt_canon_dir"

run_log_dir="$script_dir/run_log"
mkdir -p "$run_log_dir"

status_dir="$run_log_dir/status"
review_dir="$run_log_dir/review_failures"
mkdir -p "$status_dir" "$review_dir"

master_log="$run_log_dir/run_$(date +%Y%m%d_%H%M%S).log"
failed_index="$run_log_dir/failed_trials.txt"
success_index="$run_log_dir/succeeded_trials.txt"
touch "$failed_index" "$success_index"

corpus_jsonl="$script_dir/../dataset/clinical_trial/sigir/corpus.jsonl"
if [[ "$MODE" == "all" ]]; then
  [[ -f "$corpus_jsonl" ]] || { echo "ERROR: corpus not found: $corpus_jsonl" >&2; exit 2; }
fi

list_entity_summaries() {
  local tid="$1"
  shopt -s nullglob
  local files=(
    "$mbench_logs_dir/${tid}_disease_link_filter_summary.json"
    "$mbench_logs_dir/${tid}"*_disease_link_filter_summary.json
  )
  shopt -u nullglob
  for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue
    basename "$f"
  done | awk 'NF' | sort -u
}

list_build_summaries() {
  local tid="$1"
  shopt -s nullglob
  local files=( "$mirror_dir/${tid}"*_disease_link_filter_summary.json )
  shopt -u nullglob
  for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue
    basename "$f"
  done | awk 'NF' | sort -u
}

build_complete_wrt_entity_logs() {
  local tid="$1"
  local expected actual
  expected="$(list_entity_summaries "$tid" || true)"
  actual="$(list_build_summaries "$tid" || true)"

  [[ -n "$expected" ]] || return 1

  while IFS= read -r need; do
    [[ -z "$need" ]] && continue
    if ! grep -Fxq "$need" <<<"$actual"; then
      return 1
    fi
  done <<<"$expected"

  return 0
}

mirror_from_entity_logs() {
  local tid="$1"
  shopt -s nullglob
  local files=(
    "$mbench_logs_dir/${tid}_disease_link_filter_summary.json"
    "$mbench_logs_dir/${tid}"*_disease_link_filter_summary.json
  )
  shopt -u nullglob

  local matched=0
  for src in "${files[@]}"; do
    [[ -f "$src" ]] || continue
    cp -f "$src" "$mirror_dir/"
    echo "  ✓ mirrored: $(basename "$src")"
    matched=1
  done

  if [[ $matched -eq 0 ]]; then
    echo "  ⚠︎ nothing to mirror from entity_logs for $tid"
  fi
}

should_skip_compile() {
  local tid="$1"
  [[ $FORCE -eq 0 ]] || return 1

  if build_complete_wrt_entity_logs "$tid"; then
    return 0
  fi

  local expected
  expected="$(list_entity_summaries "$tid" || true)"
  if [[ -n "$expected" ]]; then
    echo "  ↪ build incomplete but entity_logs present; mirroring missing summaries..."
    mirror_from_entity_logs "$tid" >/dev/null 2>&1 || true
    if build_complete_wrt_entity_logs "$tid"; then
      return 0
    fi
    return 1
  fi

  local actual
  actual="$(list_build_summaries "$tid" || true)"
  if [[ -n "$actual" ]]; then
    return 0
  fi

  return 1
}

latest_checkpoint_for_trial() {
  local tid="$1"
  local latest=""
  shopt -s nullglob
  local candidates=(
    "$script_dir/checkpoints/${tid}"_*.json
    "$script_dir/checkpoints_target_disease/preproc/${tid}"_preproc.chkpt.json
    "$script_dir/checkpoints_target_disease/canon/${tid}"_canon.chkpt.json
  )
  shopt -u nullglob

  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo ""
    return 0
  fi

  latest="$(ls -1t "${candidates[@]}" 2>/dev/null | head -n 1 || true)"
  echo "$latest"
}

record_success() {
  local tid="$1"
  local log_path="$2"

  rm -f "$status_dir/${tid}.failed" "$review_dir/${tid}.needs_review" "$review_dir/${tid}.failed.json" 2>/dev/null || true
  : > "$status_dir/${tid}.ok"

  echo "$tid" >> "$success_index"
  echo "{\"trial_id\":\"$tid\",\"status\":\"ok\",\"log_path\":\"$log_path\",\"timestamp\":\"$(date -Is)\"}" \
    > "$status_dir/${tid}.ok.json"
}

record_failure() {
  local tid="$1"
  local log_path="$2"
  local rc="$3"
  local attempts="$4"

  local last_ckpt
  last_ckpt="$(latest_checkpoint_for_trial "$tid")"

  local entity_summaries build_summaries
  entity_summaries="$(list_entity_summaries "$tid" | jq -R . | jq -s . 2>/dev/null || printf '[]')"
  build_summaries="$(list_build_summaries "$tid" | jq -R . | jq -s . 2>/dev/null || printf '[]')"

  : > "$status_dir/${tid}.failed"
  : > "$review_dir/${tid}.needs_review"
  echo "$tid" >> "$failed_index"

  cat > "$review_dir/${tid}.failed.json" <<EOF
{
  "trial_id": $(printf '%s' "$tid" | jq -R .),
  "timestamp": $(date -Is | jq -R .),
  "exit_code": $rc,
  "attempts_exhausted": $attempts,
  "log_path": $(printf '%s' "$log_path" | jq -R .),
  "last_checkpoint": $(printf '%s' "$last_ckpt" | jq -R .),
  "entity_log_summaries": $entity_summaries,
  "build_summaries": $build_summaries
}
EOF
}

run_one() {
  local tid="$1"
  local log_path="$run_log_dir/${tid}.log"
  local canon_ckpt_path="$ckpt_canon_dir/${tid}_canon.chkpt.json"

  echo "[START] $tid"

  {
    echo "============================================================"
    echo "TRIAL_ID: $tid"
    echo "START:   $(date -Is)"
    echo "FORCE:   $FORCE"
    echo "ARGS:    ${PY_ARGS[*]-}"
    echo "============================================================"

    if should_skip_compile "$tid"; then
      echo "RESUME: skip compile for $tid (already complete in build or repaired via mirroring)."
      echo "END: $(date -Is) STATUS: SKIPPED"
      exit 0
    fi

    if [[ $FORCE -eq 1 && -f "$canon_ckpt_path" ]]; then
      rm -f "$canon_ckpt_path"
      echo "Removed old CANON checkpoint: $canon_ckpt_path"
    fi

    local attempt=1
    local rc=0
    while (( attempt <= MAX_RETRIES )); do
      echo "▶︎ $tid (attempt ${attempt}/${MAX_RETRIES})"

      set +e
      "$PYTHON" -u "$entry_py" "$tid" "${PY_ARGS[@]}"
      rc=$?
      set -e

      if [[ $rc -eq 0 ]]; then
        echo "✔︎ $tid"
        echo "↪ mirroring disease summaries for $tid..."
        mirror_from_entity_logs "$tid" || true
        echo "END: $(date -Is) STATUS: OK"
        exit 0
      fi

      echo "✖︎ $tid failed (exit $rc)"
      if (( attempt < MAX_RETRIES )); then
        echo "↻ retry in ${RETRY_DELAY}s"
        sleep "$RETRY_DELAY"
      fi
      ((attempt++))
    done

    echo "END: $(date -Is) STATUS: FAILED"
    exit "$rc"
  } >>"$log_path" 2>&1

  local final_rc=$?
  if [[ $final_rc -eq 0 ]]; then
    record_success "$tid" "$log_path"
    echo "[OK]    $tid"
  else
    record_failure "$tid" "$log_path" "$final_rc" "$MAX_RETRIES"
    echo "[FAIL]  $tid  -> flagged for review"
  fi
}

export -f run_one
export -f list_entity_summaries list_build_summaries build_complete_wrt_entity_logs
export -f mirror_from_entity_logs should_skip_compile latest_checkpoint_for_trial
export -f record_success record_failure
export PYTHON MAX_RETRIES RETRY_DELAY FORCE
export script_dir entry_py mbench_logs_dir mirror_dir ckpt_canon_dir run_log_dir status_dir review_dir failed_index success_index

PY_ARGS_DELIM=$'\n'
PY_ARGS_ENV="$(printf "%s${PY_ARGS_DELIM}" "${PY_ARGS[@]-}")"
export PY_ARGS_ENV PY_ARGS_DELIM

rehydrate_py_args() {
  mapfile -t PY_ARGS < <(printf "%s" "${PY_ARGS_ENV-}" | sed '/^$/d' || true)
}
export -f rehydrate_py_args

xargs_worker() {
  rehydrate_py_args
  run_one "$1"
}
export -f xargs_worker

get_all_trial_ids() {
  "$PYTHON" - <<PY
import json
path = r"""$corpus_jsonl"""
seen = set()
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        tid = obj.get("_id")
        if isinstance(tid, str) and tid and tid not in seen:
            seen.add(tid)
            print(tid)
PY
}

echo "Log dir:  $run_log_dir" | tee -a "$master_log"
echo "Master:   $master_log"  | tee -a "$master_log"
echo "Parallel: $PARALLEL"    | tee -a "$master_log"
echo "Mode:     $MODE"        | tee -a "$master_log"
echo "Force:    $FORCE"       | tee -a "$master_log"
echo "PY_ARGS:  ${PY_ARGS[*]-}" | tee -a "$master_log"

if [[ "$MODE" == "single" ]]; then
  [[ -n "$TRIAL_ID" ]] || usage
  run_one "$TRIAL_ID" || true
  echo "Done. See $run_log_dir/${TRIAL_ID}.log" | tee -a "$master_log"
  exit 0
fi

tmp_ids="$(mktemp)"
get_all_trial_ids > "$tmp_ids"
num_ids="$(wc -l < "$tmp_ids" | tr -d ' ')"
echo "Found $num_ids trial IDs in corpus." | tee -a "$master_log"

xargs -n 1 -P "$PARALLEL" -I {} bash -lc 'xargs_worker "$@"' _ {} < "$tmp_ids" \
  >>"$master_log" 2>&1 || true

echo "All done."
echo "Per-trial logs: $run_log_dir/"
echo "Master log:     $master_log"
echo "Failures:       $review_dir/"
rm -f "$tmp_ids"