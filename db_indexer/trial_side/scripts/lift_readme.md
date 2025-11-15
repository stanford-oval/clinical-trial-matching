python ontology_lifter.py \
  --db ${SATIR_DATA_ROOT}/build_old/trial.db \
  --canon-dir ${SATIR_DATA_ROOT}/build/canon \
  --minified-canon-dir ${SATIR_DATA_ROOT}/build/minified_canon \
  --rebuild-var2concept



SNOWSTORM_BASE=http://localhost:8080 \
python hop_policy_decider_lineage_mgsr.py \
  --db ${SATIR_DATA_ROOT}/build_old/trial.db \
  --decide --missing-only --dry-run