#!/bin/bash
# Reproduce all accuracy + counterfactual results from scratch.
#
# Stages (see REPRODUCE_FROM_SCRATCH.md for details):
#   1   SMT atom mining (cmsrc batch, ~4-8 hr, ~$50)
#   2   Arbiter (~30 min, ~$10)
#   3   Per-system verdicts (aegis, v5, tg, shah) (~1 hr, ~$30)
#   4   Verbalize AEGIS (~15 min, ~$8)
#   5   5-judge gold + bootstrap (~45 min, ~$40)
#   6   Meta-fusion (~30 min, ~$25)
#   7   Accuracy mbench (FP/FN inspection) (~5 min)
#   8   Counterfactual ablation (3 cells × 6 systems) (~90 min, ~$200)
#   9   Counterfactual mbench (~5 min)
#   10  Clinician audit sample (K=7, cell 3, ~5 min, ~$3)
#
# Usage:
#   bash scripts/reproduce_all.sh                    # full run
#   bash scripts/reproduce_all.sh --from 4           # skip stages 1-3
#   bash scripts/reproduce_all.sh --through 7        # stop after stage 7
#   bash scripts/reproduce_all.sh --only 8           # run only stage 8
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

# Defaults
FROM_STAGE=1
THROUGH_STAGE=10
PY="${PY:-python3}"
CMSRC_DIR="${CMSRC_DIR:-$ROOT/../TrialGPT-SMT/cmsrc}"
WORKERS="${WORKERS:-16}"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)    FROM_STAGE="$2"; shift 2 ;;
    --through) THROUGH_STAGE="$2"; shift 2 ;;
    --only)    FROM_STAGE="$2"; THROUGH_STAGE="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

set -a; source .env 2>/dev/null || true; set +a
: "${OPENAI_ENDPOINT:?OPENAI_ENDPOINT must be set in .env}"

stage() {
  local n="$1"; shift
  if [ "$n" -lt "$FROM_STAGE" ] || [ "$n" -gt "$THROUGH_STAGE" ]; then
    echo "[skip stage $n] $*"
    return 0
  fi
  echo
  echo "==================================================================="
  echo "[STAGE $n] $* (started $(date +%H:%M:%S))"
  echo "==================================================================="
}

# ── Stage 1: SMT atom mining ───────────────────────────────────────────────
stage 1 "SMT atom mining via cmsrc batch driver"
if [ "$FROM_STAGE" -le 1 ] && [ "$THROUGH_STAGE" -ge 1 ]; then
  if [ ! -d "$CMSRC_DIR" ]; then
    echo "ERROR: CMSRC_DIR=$CMSRC_DIR not found."
    echo "Stage 1 re-mines the whole corpus with the BATCH driver, which is not"
    echo "vendored. The single-pair matcher itself is vendored at verdict/engine/"
    echo "-- use 'verdict run TRIAL PATIENT' for one pair without this checkout."
    echo "To re-mine the corpus, set CMSRC_DIR to your cmsrc checkout."
    exit 1
  fi
  ( cd "$CMSRC_DIR" && "$PY" batch_match_from_eval_union.py \
      --out-root "$ROOT/experiments/53_v2_full/cmsrc_out_REMINE_v9_full" \
      --prompt-root "$ROOT/experiments/53_v2_full/inputs/prompt_root" \
      --data-root "$ROOT/dataset/clinical_trial" \
      --workers "$WORKERS" )
fi

# ── Stage 2: Arbiter ───────────────────────────────────────────────────────
stage 2 "Arbiter: chart-grounded atom overrides"
if [ "$FROM_STAGE" -le 2 ] && [ "$THROUGH_STAGE" -ge 2 ]; then
  "$PY" matchers/systems/aegis/aegis_arbiter.py \
      --mine-root experiments/53_v2_full/cmsrc_out_REMINE_v9_full \
      --out-cache experiments/53_v2_full/arbiter_cache \
      --workers "$WORKERS"
fi

# ── Stage 3: Per-system verdicts ───────────────────────────────────────────
stage 3 "Per-system verdicts (aegis, single_shot_llm, trialgpt, shahlab)"
if [ "$FROM_STAGE" -le 3 ] && [ "$THROUGH_STAGE" -ge 3 ]; then
  "$PY" matchers/systems/aegis/run.py
  "$PY" matchers/systems/single_shot_llm/run.py --variant V5_TWO_STEP --workers 10
  "$PY" matchers/systems/trialgpt/run.py --workers 10
  "$PY" matchers/systems/shahlab/run.py --workers 10
fi

# ── Stage 4: Verbalize AEGIS ───────────────────────────────────────────────
stage 4 "Verbalize AEGIS symbolic → NL"
if [ "$FROM_STAGE" -le 4 ] && [ "$THROUGH_STAGE" -ge 4 ]; then
  "$PY" verbalizer/run_aegis_v9_arbiter_verbalize.py --workers "$WORKERS"
fi

# ── Stage 5: 5-judge gold + bootstrap ──────────────────────────────────────
stage 5 "5-judge gold ensemble + bootstrap CI"
if [ "$FROM_STAGE" -le 5 ] && [ "$THROUGH_STAGE" -ge 5 ]; then
  "$PY" experiments/accuracy/scripts/run_5judges_verbalized.py --workers "$WORKERS"
  "$PY" experiments/accuracy/scripts/build_refined_gold_from_5judges.py
  "$PY" experiments/accuracy/scripts/bootstrap_ci.py
fi

# ── Stage 6: Meta-fusion ───────────────────────────────────────────────────
stage 6 "Meta-fusion (balanced/precision/recall)"
if [ "$FROM_STAGE" -le 6 ] && [ "$THROUGH_STAGE" -ge 6 ]; then
  "$PY" experiments/accuracy/scripts/run_meta_fusion.py --mode balanced  --workers 12
  "$PY" experiments/accuracy/scripts/run_meta_fusion.py --mode precision --workers 12
  "$PY" experiments/accuracy/scripts/run_meta_fusion.py --mode recall    --workers 12
fi

# ── Stage 7: Accuracy mbench ───────────────────────────────────────────────
stage 7 "Accuracy mbench (FP/FN inspection drilldowns)"
if [ "$FROM_STAGE" -le 7 ] && [ "$THROUGH_STAGE" -ge 7 ]; then
  "$PY" experiments/accuracy/scripts/build_aegis_fpfn_mbench_balanced.py
  "$PY" experiments/accuracy/scripts/build_aegis_v5_mbench.py
  "$PY" experiments/accuracy/scripts/build_arbiter_v2_mbench.py
fi

# ── Stage 8: Counterfactual ablation ───────────────────────────────────────
stage 8 "Counterfactual self-faithfulness (3 cells × 6 systems × 285 pairs)"
if [ "$FROM_STAGE" -le 8 ] && [ "$THROUGH_STAGE" -ge 8 ]; then
  bash experiments/counterfactual/05_self_faithfulness/ablation_full_corpus/launch_all.sh
  "$PY" experiments/counterfactual/05_self_faithfulness/ablation_full_corpus/summarize_3cell.py
fi

# ── Stage 9: Counterfactual mbench ─────────────────────────────────────────
stage 9 "Counterfactual mbench (per-cell per-system drilldown)"
if [ "$FROM_STAGE" -le 9 ] && [ "$THROUGH_STAGE" -ge 9 ]; then
  "$PY" experiments/counterfactual/05_self_faithfulness/ablation_full_corpus/build_mbench_3cell.py
fi

# ── Stage 10: Clinician audit ──────────────────────────────────────────────
stage 10 "Clinician audit sample (K=7, cell 3, 118 cells)"
if [ "$FROM_STAGE" -le 10 ] && [ "$THROUGH_STAGE" -ge 10 ]; then
  "$PY" experiments/clinician_validation/build_cf_audit_K7_v2.py
  "$PY" experiments/clinician_validation/inject_simclin_into_topics.py
  "$PY" experiments/clinician_validation/claude_simclin_judgments.py
fi

echo
echo "==================================================================="
echo "DONE (finished $(date +%H:%M:%S))"
echo "==================================================================="
echo
echo "Outputs:"
echo "  Accuracy:   experiments/accuracy/{results.md, bootstrap_ci.json}"
echo "  Acc mbench: experiments/accuracy/inspection/*"
echo "  CF:         experiments/counterfactual/05_self_faithfulness/ablation_full_corpus/summary_3cell.csv"
echo "  CF mbench:  experiments/counterfactual/05_self_faithfulness/mbench_3cell/"
echo "  Clinician:  experiments/clinician_validation/cf_audit_K7_cell3.json"
echo "              clinical-trial-annotation-frontend/private/clinician_review.json"
