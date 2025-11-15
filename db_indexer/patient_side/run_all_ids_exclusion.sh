#!/usr/bin/env bash
set -euo pipefail

IDS_FILE="${1:-ids_sigir.txt}"

PY="${SATIR_DATA_ROOT}/irsrc/patient_side/patient_coded_fact_entailment_pipeline_diagnose_classifed.py"

BUILD_ROOT="${SATIR_DATA_ROOT}/patient_build_test_exclusion"
CERTAIN_ROOT="${SATIR_DATA_ROOT}/patient_build_sigir_exclusion/patient_facts_export"
SOURCE_ROOT="${SATIR_DATA_ROOT}/patient_build_sigir/patient_coded_results"


PASSES=3
SIDE="exclusion"

# 逐行读 id（跳过空行和 # 注释）
while IFS= read -r ID; do
  [[ -z "${ID// }" ]] && continue
  [[ "${ID:0:1}" == "#" ]] && continue

  echo "============================================================"
  echo "[RUN] patient=${ID} side=${SIDE}"
  echo "============================================================"

  CERTAIN_FILE="${CERTAIN_ROOT}/${ID}_exclusion/exclusion/canonical.jsonl"

  SRC1="${SOURCE_ROOT}/${ID}_exclusion/diagnosis.jsonl"
  SRC2="${SOURCE_ROOT}/${ID}_exclusion/embedding_search_other_candidate_canonical.jsonl"

  python "$PY" \
    --build-root "$BUILD_ROOT" \
    --certain-file-path "$CERTAIN_FILE" \
    --passes "$PASSES" \
    --side "$SIDE" \
    --patient "$ID" \
    --source-jsonl "$SRC1" \
    --source-jsonl "$SRC2" 

  echo "[DONE] ${ID}"
done < "$IDS_FILE"
