#!/bin/bash
# Common prelude sourced by all reproduction scripts.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"
[ -f .env ] && set -a && source .env && set +a
: "${OPENAI_ENDPOINT:?OPENAI_ENDPOINT must be set in .env}"
export PY="${PY:-python3}"
export DEFAULT_PAIRS="${DEFAULT_PAIRS:-$ROOT/experiments/clinician_validation/repro50_pairs.txt}"
export OUT_DIR="${OUT_DIR:-$ROOT/experiments/clinician_validation/repro_runs}"
mkdir -p "$OUT_DIR"
