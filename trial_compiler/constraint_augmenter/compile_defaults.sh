#!/usr/bin/env bash
# compile_defaults.sh — capped parallel runner for defaults generation (per-trial)
# Runs your generate_default_values.py per trial ID with retries and bounded parallelism.
# Safe on CI/pipes/tmux/VSCode: the cancel watcher only starts if an interactive /dev/tty exists.
set -euo pipefail

: "${PYTHON:=python3}"
: "${PY_ENTRY:=generate_default_values.py}"   # entrypoint Python file
: "${JOBS:=24}"                               # outer parallelism (shell-level)
: "${MAX_RETRIES:=8}"
: "${RETRY_DELAY:=3}"

# Where the *source* canonical JSONs live:
: "${CANON_SRC_DIR:="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../build/canon"}"
: "${OUTPUT_DIR:="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../build/default_vars"}"
: "${MBENCH_DIR:="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/./mbench"}"

# Prompt/model/engine passed to Python (and used by queue script for up-to-date checks)
: "${PROMPT_PATH:="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/prompts/decide_default_values.prompt"}"
: "${MODEL_NAME:=gpt-4.1}"
: "${ENDPOINT:="${OPENAI_ENDPOINT:-}"}"
: "${FALLBACK_ENDPOINT:=}"

# Inner workers: your Python also parallelizes; default to 1 to avoid oversubscription
: "${INNER_WORKERS:=1}"

READ_STDIN=0
LOG_DIR=""
PASS_SKIP_EXISTING=0
PASS_SKIP_UPTODATE=0

# ---- cancellation handling ----
CANCELLED=0
CANCEL_KEY="${CANCEL_KEY:-q}"
watcher_pid=""

on_cancel() {
  CANCELLED=1
  echo "↯ Caught signal; stopping new work and terminating workers..." >&2
  # Prevent re-entrancy
  trap - INT TERM
  # Send TERM to the whole process group (this shell + background jobs)
  kill -TERM -$$ 2>/dev/null || true
  # Give workers a brief chance to exit gracefully
  sleep 1
  # Ensure stragglers are gone
  kill -KILL -$$ 2>/dev/null || true
}
trap 'on_cancel' INT TERM

usage() {
  cat <<EOF
USAGE: $0 [-j N|--jobs N] [--log-dir DIR] [--stdin]
          [--skip-existing] [--skip-uptodate]
          [--model NAME] [--endpoint URL] [--fallback-endpoint URL]
          [--prompt-path FILE] [--canon-src DIR] [--output-dir DIR] [--mbench-dir DIR]
          [--inner-workers N]
          <TRIAL_ID...>

ENV toggles:
  DISABLE_CANCEL_WATCHER=1   # disable 'press q to cancel' helper
  CANCEL_KEY=q               # change the cancel key for the watcher
EOF
  exit 1
}

ARGS=()
while (( "$#" )); do
  case "$1" in
    -j|--jobs) JOBS="$2"; shift 2 ;;
    --jobs=*)  JOBS="${1#*=}"; shift ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --log-dir=*) LOG_DIR="${1#*=}"; shift ;;
    --stdin) READ_STDIN=1; shift ;;
    --skip-existing) PASS_SKIP_EXISTING=1; shift ;;
    --skip-uptodate) PASS_SKIP_UPTODATE=1; shift ;;
    --model) MODEL_NAME="$2"; shift 2 ;;
    --model=*) MODEL_NAME="${1#*=}"; shift ;;
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --endpoint=*) ENDPOINT="${1#*=}"; shift ;;
    --fallback-endpoint) FALLBACK_ENDPOINT="$2"; shift 2 ;;
    --fallback-endpoint=*) FALLBACK_ENDPOINT="${1#*=}"; shift ;;
    --prompt-path) PROMPT_PATH="$2"; shift 2 ;;
    --prompt-path=*) PROMPT_PATH="${1#*=}"; shift ;;
    --canon-src) CANON_SRC_DIR="$2"; shift 2 ;;
    --canon-src=*) CANON_SRC_DIR="${1#*=}"; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
    --mbench-dir) MBENCH_DIR="$2"; shift 2 ;;
    --mbench-dir=*) MBENCH_DIR="${1#*=}"; shift ;;
    --inner-workers) INNER_WORKERS="$2"; shift 2 ;;
    --inner-workers=*) INNER_WORKERS="${1#*=}"; shift ;;
    -h|--help) usage ;;
    -* ) echo "ERROR: unknown option $1" >&2; usage ;;
    * ) ARGS+=("$1"); shift ;;
  esac
done

[[ ${#ARGS[@]} -gt 0 || $READ_STDIN -eq 1 ]] || usage
case "$JOBS" in ''|*[!0-9]*|0) echo "ERROR: -j must be positive int" >&2; exit 2 ;; esac
[[ -n "$LOG_DIR" ]] && mkdir -p "$LOG_DIR"
command -v "$PYTHON" >/dev/null || { echo "ERROR: python not found: $PYTHON" >&2; exit 2; }
[[ -f "$PY_ENTRY" ]] || { echo "ERROR: entry not found: $PY_ENTRY" >&2; exit 2; }
[[ -d "$CANON_SRC_DIR" ]] || { echo "ERROR: canon-src not found: $CANON_SRC_DIR" >&2; exit 2; }

# ---- robust cancel watcher (only if interactive and /dev/tty is readable) ----
start_cancel_watcher() {
  [[ -n "${DISABLE_CANCEL_WATCHER:-}" ]] && return 0
  [[ -t 0 && -t 1 && -r /dev/tty ]] || return 0

  (
    echo "Press '${CANCEL_KEY}' to cancel…"
    exec 3</dev/tty || exit 0
    while true; do
      if IFS= read -r -n 1 -t 1 ch <&3 2>/dev/null; then
        [[ "$ch" == "$CANCEL_KEY" ]] && { echo; kill -INT $$; break; }
      fi
      # exit if parent died
      kill -0 $PPID 2>/dev/null || exit 0
    done
  ) & watcher_pid=$!
}

stop_cancel_watcher() {
  [[ -n "${watcher_pid:-}" ]] && kill "$watcher_pid" 2>/dev/null || true
}

start_cancel_watcher
trap 'stop_cancel_watcher' EXIT

# ---- per-trial task runner with retries ----
# Create a temp staging root for per-trial canon directories
stage_root="$(mktemp -d "${TMPDIR:-/tmp}/defaults_stage.XXXXXX")"
cleanup_stage() { rm -rf "$stage_root" 2>/dev/null || true; }
trap 'cleanup_stage' EXIT INT TERM

run_task() {
  local trial_id="$1"
  (( CANCELLED )) && return 130

  # Stage minimal canon dir: only inclusion files for this trial
  local workdir="$stage_root/$trial_id"
  local canon_dir="$workdir/canon"
  mkdir -p "$canon_dir"

  local linked=0
  while IFS= read -r src; do
    [[ -z "$src" ]] && continue
    ln -s "$src" "$canon_dir/$(basename "$src")"
    linked=1
  done < <(compgen -G "$CANON_SRC_DIR/${trial_id}"'*_inclusion_canonical_variables.json' || true)

  if (( ! linked )); then
    echo "⚠︎ No inclusion canonical for ${trial_id}; skipping." >&2
    return 0
  fi

  local sink="/dev/null"
  [[ -n "$LOG_DIR" ]] && sink="${LOG_DIR%/}/${trial_id}.log"

  local attempt=1 rc=0
  while (( attempt <= MAX_RETRIES )); do
    (( CANCELLED )) && return 130

    if (( MAX_RETRIES > 1 )); then
      echo "▶︎ Starting ${trial_id} — attempt ${attempt}/${MAX_RETRIES}"
    else
      echo "▶︎ Starting ${trial_id}"
    fi

    cmd=( "$PYTHON" -u "$PY_ENTRY"
          --canon-dir "$canon_dir"
          --output-dir "$OUTPUT_DIR"
          --prompt-path "$PROMPT_PATH"
          --model "$MODEL_NAME"
          --mbench-dir "$MBENCH_DIR"
          --workers "$INNER_WORKERS"
        )
    [[ -n "$ENDPOINT"          ]] && cmd+=( --endpoint "$ENDPOINT" )
    [[ -n "$FALLBACK_ENDPOINT" ]] && cmd+=( --fallback-endpoint "$FALLBACK_ENDPOINT" )
    (( PASS_SKIP_EXISTING )) && cmd+=( --skip-existing )
    (( PASS_SKIP_UPTODATE )) && cmd+=( --skip-uptodate )

    if (( attempt == 1 )); then
      if "${cmd[@]}" >"$sink" 2>&1 </dev/null; then
        echo "✔︎ Done ${trial_id}"
        rc=0
      else
        rc=$?
      fi
    else
      if "${cmd[@]}" >>"$sink" 2>&1 </dev/null; then
        echo "✔︎ Done ${trial_id} (after ${attempt} attempts)"
        rc=0
      else
        rc=$?
      fi
    fi

    (( rc == 0 )) && return 0
    (( attempt < MAX_RETRIES )) || break
    echo "↻ Retry ${trial_id} in ${RETRY_DELAY}s (exit $rc; log: ${sink})" >&2
    sleep "$RETRY_DELAY"
    ((attempt++))
  done

  echo "✖︎ FAILED ${trial_id} — see ${sink} (exit $rc)" >&2
  return $rc
}

# ---- portable FIFO pool (no wait -n) ----
declare -a pids=() tasks=()
inflight=0
fail=0

reap_one_finished() {
  local i pid rc
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    [[ -z "$pid" ]] && continue
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then rc=0; else rc=$?; fail=1; fi
      pids[$i]="" ; (( inflight>0 )) && ((inflight--))
      return 0
    fi
  done
  return 1
}

wait_for_slot(){
  while (( inflight >= JOBS )); do
    (( CANCELLED )) && return 0
    reap_one_finished || sleep 0.1
  done
}

declare -a QUEUE=("${ARGS[@]}")
qhead=0

refill_from_stdin(){
  (( READ_STDIN )) || return 0
  (( CANCELLED )) && return 0
  local line
  if IFS= read -r -t 0.01 line; then
    [[ -n "$line" ]] && QUEUE+=( "$line" )
    while IFS= read -r -t 0 line; do [[ -n "$line" ]] && QUEUE+=( "$line" ); done
  fi
}

# ---- main scheduling loop ----
while :; do
  refill_from_stdin

  while (( qhead < ${#QUEUE[@]} )); do
    (( CANCELLED )) && break
    wait_for_slot
    (( CANCELLED )) && break

    tid="${QUEUE[$qhead]}"; ((qhead++))
    run_task "$tid" & pids+=( "$!" ); tasks+=( "$tid" ); ((inflight++))

    reap_one_finished || true
    refill_from_stdin
  done

  (( CANCELLED )) && break

  if (( inflight > 0 )); then
    reap_one_finished || sleep 0.1
    continue
  fi

  (( READ_STDIN )) || break
  refill_from_stdin
  (( qhead < ${#QUEUE[@]} )) && continue
  break
done

# ---- drain remaining workers ----
for i in "${!pids[@]}"; do
  [[ -z "${pids[$i]}" ]] && continue
  if ! wait "${pids[$i]}"; then fail=1; fi
done

# If cancelled, return SIGINT-style exit code
if (( CANCELLED )); then
  echo "↯ Cancelled by user. Exiting." >&2
  exit 130
fi

exit $fail
