# run single multi-cohort trial, mirror to build, each subcohort saved in separate files (saved in ../build1 for debug)
```
./compile_disease_simple.sh NCT00000665 --force
```

# run dsrc from corpus.jsonl, allow multiple subcohort works on linux

```
# skip trials already existed in build folder (resume):
./compile_disease_from_json_subcohort.sh --all --parallel 16
```

```
# force to overwrite:
./compile_disease_from_json_subcohort.sh --all --parallel 16 --force
```


# old version, doesn't support subcohort build
./compile_disease_from_tsv.sh -n 24 -j 6 \
  ../dataset/clinical_trial/splits_query_then_drop/sigir_train.tsv -- \
  --side both --stop-after canon

result: /TrialGPT-SMT/build/disease, "final_selected_concept_by_disease"
mbench: /TrialGPT-SMT/dsrc/mbench
run_log: /TrialGPT-SMT/dsrc/run_logs


