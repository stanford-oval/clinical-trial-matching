#!/usr/bin/env bash
# compile_diseases_from_jsonl.sh — read IDs from JSONL ("_id"), skip built (CANON), run compile_disease.py in parallel
set -Eeuo pipefail

# -------- defaults --------
PYTHON="${PYTHON:-python3}"
JOBS="${JOBS:-24}"            # -j/--jobs 覆盖
LIMIT=""                     # -n/--limit 限制数量（跳过已建后再截断）
JSONL=""                     # 必填：JSONL 路径（每行一个 JSON，包含 "_id"）
MAX_RETRIES="${MAX_RETRIES:-3}"
RETRY_DELAY="${RETRY_DELAY:-3}"

# 剩余参数（在 "--" 之后）透传给 compile_disease.py，例如：--side both --stop-after canon --no-prompts ...
PY_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [-j N|--jobs N] [-n N|--limit N] <path/to/file.jsonl> -- [args passed to compile_disease.py...]

Example:
  $0 -n 1000 -j 8 ../dataset/clinical_trial/splits_query_then_drop/sigir_train.jsonl -- --side both --stop-after canon
EOF
  exit 1
}

# -------- parse args --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -j|--jobs)   JOBS="$2"; shift 2 ;;
    --jobs=*)    JOBS="${1#*=}"; shift ;;
    -n|--limit)  LIMIT="$2"; shift 2 ;;
    --limit=*)   LIMIT="${1#*=}"; shift ;;
    --)
      shift
      PY_ARGS+=("$@")
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2; usage ;;
    *)
      if [[ -z "$JSONL" && -f "$1" ]]; then
        JSONL="$1"; shift
      else
        echo "Unexpected arg: $1" >&2; usage
      fi
      ;;
  esac
done

[[ -n "$JSONL" ]] || usage
case "$JOBS" in ''|*[!0-9]*|0) echo "ERROR: -j/--jobs must be a positive integer" >&2; exit 2 ;; esac
command -v "$PYTHON" >/dev/null || { echo "ERROR: python not found: $PYTHON" >&2; exit 2; }

# -------- paths --------
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
entry_py="$script_dir/compile_disease.py"
[[ -f "$entry_py" ]] || { echo "ERROR: entry not found: $entry_py" >&2; exit 2; }

# mbench 源目录 + 目标镜像目录
mbench_logs_dir="$script_dir/mbench/entity_mbench/entity_logs"
mirror_dir="$script_dir/../build/disease"
mkdir -p "$mirror_dir"

# compile_disease.py 的 ckpt_dir 默认相对 CWD 为 "checkpoints_target_disease"
# 这里按“脚本所在目录”为基准检查 CANON 是否已完成
ckpt_canon_dir="$script_dir/checkpoints_target_disease/canon"

is_built() {
  local id="$1"
  [[ -e "$ckpt_canon_dir/${id}_canon.chkpt.json" ]]
}

# -------- read IDs from JSONL (每行一个 JSON，字段 "_id") --------
id_filter() {
  "$PYTHON" - "$JSONL" <<'PY'
import sys, json

if len(sys.argv) < 2:
    sys.exit("missing JSONL path")
path = sys.argv[1]

seen = set()
with open(path, encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        tid = str(obj.get("_id","")).strip()
        if tid and tid not in seen:
            seen.add(tid)
            print(tid)
PY
}

# -------- prepare ID queue (skip built, then apply LIMIT) --------
ids=()
while IFS= read -r id; do
  if [[ -d "$ckpt_canon_dir" ]] && is_built "$id"; then
    continue
  fi
  ids+=("$id")
  if [[ -n "$LIMIT" && ${#ids[@]} -ge $LIMIT ]]; then
    break
  fi
done < <(id_filter)

if [[ ${#ids[@]} -eq 0 ]]; then
  echo "All trials in '$JSONL' appear to have CANON checkpoints in '$ckpt_canon_dir' (nothing to do)." >&2
  exit 0
fi

echo "Queuing ${#ids[@]} trial(s) from '$JSONL' (after skipping CANON-built)..." >&2

# -------- per-task runner with retries --------
run_one() {
  local id="$1"
  local attempt=1 rc=0

  while (( attempt <= MAX_RETRIES )); do
    if (( MAX_RETRIES > 1 )); then
      echo "▶︎ ${id} — attempt ${attempt}/${MAX_RETRIES}"
    else
      echo "▶︎ ${id}"
    fi

    # 不做任何外层重定向：让 compile_disease.py 自己写 run_logs/
    if "$PYTHON" -u "$entry_py" "$id" "${PY_ARGS[@]}"; then
      echo "✔︎ ${id}"

      # 成功后复制 mbench 的 summary 到 build/disease/
      local src="${mbench_logs_dir}/${id}_disease_link_filter_summary.json"
      if [[ -f "$src" ]]; then
        cp -f "$src" "$mirror_dir/" && echo "↪ mirrored: $src -> $mirror_dir/"
      else
        echo "⚠︎ summary not found for ${id}: $src" >&2
      fi
      return 0
    else
      rc=$?
    fi

    if (( attempt < MAX_RETRIES )); then
      echo "↻ retry ${id} in ${RETRY_DELAY}s (exit $rc)" >&2
      sleep "$RETRY_DELAY"
    fi
    ((attempt++))
  done

  echo "✖︎ FAILED ${id} (exit $rc)" >&2
  return $rc
}

# -------- portable pool (no wait -n) --------
declare -a pids=() tasks=()
inflight=0
fail=0
qhead=0

reap_one_finished() {
  local i pid rc
  for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    [[ -z "$pid" ]] && continue
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then rc=0; else rc=$?; fail=1; fi
      pids[$i]="" ; (( inflight>0 )) && (( inflight-- ))
      return 0
    fi
  done
  return 1
}

wait_for_slot() {
  while (( inflight >= JOBS )); do
    reap_one_finished || sleep 0.1
  done
}

cleanup() {
  local tk=()
  for i in "${!pids[@]}"; do [[ -n "${pids[$i]:-}" ]] && tk+=( "${pids[$i]}" ); done
  ((${#tk[@]})) && { kill "${tk[@]}" 2>/dev/null || true; wait "${tk[@]}" 2>/dev/null || true; }
}
trap 'echo "Aborting…"; cleanup; exit 130' INT TERM

while (( qhead < ${#ids[@]} )); do
  wait_for_slot
  id="${ids[$qhead]}"; ((qhead++))
  run_one "$id" &
  pids+=( "$!" )
  tasks+=( "$id" )
  (( inflight++ ))
  reap_one_finished || true
done

for i in "${!pids[@]}"; do
  [[ -z "${pids[$i]}" ]] && continue
  if ! wait "${pids[$i]}"; then
    rc=$?
    [[ $rc -eq 127 ]] && continue
    echo "✖︎ Task failed: ${tasks[$i]}" >&2
    fail=1
  fi
done

exit $fail
