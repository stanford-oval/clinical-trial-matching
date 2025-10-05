#!/usr/bin/env bash
# queue_defaults.sh — queue trials for the defaults generator, skipping built/up-to-date
set -Eeuo pipefail

limit=""        # -n/--limit to cap IDs (after filtering)
jsonl=""        # REQUIRED: JSONL with one object per line containing an "_id"
compile_args=()
stream=0        # --stream to feed IDs over stdin to compile_defaults.sh --stdin

# --- paths (match your repo layout; change if needed) ---
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_root="$script_dir/../build"                 # base build folder
canon_root="$script_dir/../build/canon"           # where *_canonical_variables.json live
output_dir="$script_dir/../build/default_vars"    # where defaults JSONs are written
preproc_dir="$script_dir/mbench/preproc_logs"  # where normalized manifests live

mkdir -p "$output_dir"

# Defaults for prompt/model (only used for "up-to-date" checks if you request it)
DEFAULT_PROMPT_PATH="$script_dir/prompts/decide_default_values.prompt"
DEFAULT_MODEL="gpt-4.1"

# Parse: [-n N] [--stream] <file.jsonl> -- [args for compile_defaults.sh...]
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--limit)       limit="${2:?}"; shift 2 ;;
    --stream)         stream=1; shift ;;
    --)               shift; compile_args+=("$@"); break ;;
    -*)
      # forward unknown flags to compile_defaults.sh
      compile_args+=("$1"); shift ;;
    *)
      if [[ -z "$jsonl" && -f "$1" ]]; then
        jsonl="$1"; shift
      else
        compile_args+=("$1"); shift
      fi ;;
  esac
done

if [[ -z "$jsonl" ]]; then
  echo "Usage: $0 [-n N] [--stream] <file.jsonl> -- [compile_defaults.sh args...]" >&2
  exit 1
fi

# --- read unique parent IDs from JSONL (_id) safely ---
parent_ids(){ python3 - "$jsonl" <<'PY'
import sys, json
seen=set()
with open(sys.argv[1], encoding="utf-8") as f:
    for ln in f:
        ln=ln.strip()
        if not ln: continue
        try:
            obj=json.loads(ln)
        except Exception:
            continue
        tid=str(obj.get("_id","")).strip()
        if tid and tid not in seen:
            seen.add(tid)
            print(tid)
PY
}

# --- expand to effective subcohort IDs (if available) ---
effective_ids_for_parent() {
  local parent="$1"
  local out=()
  local inc_json="$preproc_dir/${parent}_inclusion.pre.normalized.json"
  local exc_json="$preproc_dir/${parent}_exclusion.pre.normalized.json"
  if [[ -f "$inc_json" || -f "$exc_json" ]]; then
    mapfile -t out < <(python3 - "$inc_json" "$exc_json" <<'PY'
import json, sys, os
seen=set()
for p in sys.argv[1:]:
    if not p or not os.path.isfile(p): continue
    try:
        with open(p, encoding="utf-8") as f:
            rec=json.load(f)
        for tid in rec.get("effective_trial_ids", []):
            if isinstance(tid, str) and tid.strip():
                seen.add(tid.strip())
    except Exception:
        pass
print("\n".join(sorted(seen)))
PY
)
  fi
  ((${#out[@]})) || out=("$parent")
  printf '%s\n' "${out[@]}"
}

# --- test if a parent is "done" ---
# Done if *all* effective IDs have an up-to-date inclusion defaults artifact.
# We treat either "<id>_inclusion_defaults.json" or "<id>_inclusion.json" as the artifact,
# because your Python currently writes "<trial_id>_<inc_exc>.json".
is_done_parent() {
  local parent="$1"
  local prompt_path="${PROMPT_PATH:-$DEFAULT_PROMPT_PATH}"
  local model="${MODEL_NAME:-$DEFAULT_MODEL}"

  python3 - "$parent" "$canon_root" "$output_dir" "$preproc_dir" "$prompt_path" "$model" <<'PY'
import sys, os, json, hashlib
from pathlib import Path

def sha256_text(t: str) -> str:
    try: return hashlib.sha256(t.encode("utf-8")).hexdigest()
    except Exception: return "NA"

parent, canon_root, out_dir, preproc_dir, prompt_path, model = sys.argv[1:7]
canon_root, out_dir, preproc_dir = map(Path, (canon_root, out_dir, preproc_dir))

def effective_ids(parent_id):
    inc = preproc_dir / f"{parent_id}_inclusion.pre.normalized.json"
    exc = preproc_dir / f"{parent_id}_exclusion.pre.normalized.json"
    ids=set()
    for p in (inc, exc):
        if p.is_file():
            try:
                rec=json.loads(p.read_text(encoding="utf-8"))
                for tid in rec.get("effective_trial_ids", []):
                    if isinstance(tid, str) and tid.strip():
                        ids.add(tid.strip())
            except Exception:
                pass
    return sorted(ids) or [parent_id]

# Load prompt once for hash
try:
    prompt_text = Path(prompt_path).read_text(encoding="utf-8").strip()
except Exception:
    prompt_text = ""
prompt_hash = sha256_text(prompt_text)

def uptodate_for_id(eid: str) -> bool:
    # Inclusion only (your Python filters inclusion files)
    # Try two possible artifact names
    out1 = out_dir / f"{eid}_inclusion_defaults.json"
    out2 = out_dir / f"{eid}_inclusion.json"
    out_path = out1 if out1.exists() else out2
    if not out_path.exists():
        return False

    # Input canonical file (allow letter suffixes): NCT..._inclusion_canonical_variables.json
    # Prefer exact eid match, otherwise accept first glob hit for compatibility.
    import glob
    patt = str((Path(canon_root) / f"{eid}_inclusion_canonical_variables.json"))
    cand = Path(patt)
    if not cand.exists():
        # Accept * suffix variants (e.g., a/b)
        gl = glob.glob(str(Path(canon_root) / f"{eid}*_inclusion_canonical_variables.json"))
        if not gl:
            return False
        cand = Path(gl[0])

    try:
        if out_path.stat().st_mtime < cand.stat().st_mtime:
            return False
    except Exception:
        return False

    # Check mbench meta (model + prompt hash)
    meta_path = Path(preproc_dir).parent / "mbench" / eid / "mbench_meta.json"  # sibling of trial mbench? user’s Python: mbench/<trial_id>/mbench_meta.json
    if not meta_path.exists():
        # Be permissive: treat as not up-to-date; caller may still pass --skip-existing
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(meta.get("model","")) != str(model):
        return False
    if meta.get("prompt_sha256") != prompt_hash:
        return False
    return True

for eid in effective_ids(parent):
    if not uptodate_for_id(eid):
        print("NOT_DONE")
        sys.exit(0)
print("DONE")
PY
}

# Build the filtered list in order, applying limit *after* skipping
ids=()
while IFS= read -r pid; do
  [[ -z "$pid" ]] && continue
  if [[ "$(is_done_parent "$pid")" == "DONE" ]]; then
    continue
  fi
  ids+=("$pid")
  if [[ -n "$limit" && ${#ids[@]} -ge $limit ]]; then
    break
  fi
done < <(parent_ids)

if [[ ${#ids[@]} -eq 0 ]]; then
  echo "All trials in '$jsonl' look built or up-to-date in '$output_dir' (nothing to do)." >&2
  exit 0
fi

echo "Queuing ${#ids[@]} trial(s) from '$jsonl' (after skipping built/up-to-date)..." >&2

# Always hint the Python side to skip existing/uptodate as a safety net.
compile_args+=( --skip-existing --skip-uptodate )

if (( stream )); then
  added_stdin=0
  for a in "${compile_args[@]}"; do [[ $a == "--stdin" ]] && added_stdin=1; done
  (( added_stdin )) || compile_args+=( --stdin )
  printf '%s\n' "${ids[@]}" | exec "$script_dir/compile_defaults.sh" "${compile_args[@]}"
else
  exec "$script_dir/compile_defaults.sh" "${compile_args[@]}" "${ids[@]}"
fi
