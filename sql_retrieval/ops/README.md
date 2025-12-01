python compose_trial_eval.py
python eval_pr_rec_at_k.py --ground-truth ${SATIR_DATA_ROOT}/dataset/clinical_trial/sigir/qrels/test.tsv

python build_patient_trial_kits.py \
  --project-root ${SATIR_DATA_ROOT} \
  --retrieved-root retrieved_mappings \
  --out out_sigir_kits \
  --corpus ${SATIR_DATA_ROOT}/dataset/clinical_trial/sigir/corpus.jsonl \
  --queries ${SATIR_DATA_ROOT}/dataset/clinical_trial/sigir/queries.jsonl \