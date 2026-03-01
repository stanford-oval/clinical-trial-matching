# matching_batch/ — Batch orchestration for patient-trial matching

Scripts that run the `smt_matcher` pipeline over many (patient, trial) pairs
in parallel, typically driven by a retrieval evaluation's selection output.

## Scripts

### `batch_match_from_list_to_match.py`
Matches all (patient, trial) pairs enumerated in `list_to_match__*` files
produced by `sql_retrieval.ops.constraint_retrieval`. Canonical-parent
normalized: runs the parent trial once even if multiple subcohorts were
selected, and aggregates all subcohort signals.

### `batch_match_from_eval_union.py`
Matches the UNION of SMT + TrialGPT eval selections. For each patient, a
trial is included if either side (SMT "all_satisfied" OR TrialGPT "m") marked
it relevant. Runs each selected trial individually (no canonical dedup).

### `export_patient_solver_status.py`
Aggregates per-trial solver results into a per-patient summary with:
trial NCT, eligible flag, inclusion/exclusion status, unsat assertions,
and unsat core. Simple post-processing step.

## Typical workflow

```bash
# 1. Retrieve candidates
python -m sql_retrieval.ops.constraint_retrieval --db build/trial.db --out ./retrieve_out ...

# 2. Match the candidates (with optional judges)
python -m matching_batch.batch_match_from_list_to_match \
    --list-to-match-root ./retrieve_out/list_to_match__all__prevent__act \
    --out-root ./match_out \
    --enable-llm-judge --enable-trialgpt-judge \
    --parallel 8

# 3. Aggregate per-patient solver status
python -m matching_batch.export_patient_solver_status \
    --list-to-match-root ./retrieve_out/list_to_match__all__prevent__act \
    --match-out-root ./match_out \
    --out-dir ./solver_status
```

## Cache layout

Each batch run creates `<out-root>/_prompt_cache/{stage}/...` keyed by
content fingerprint. Re-runs hit the cache unless inputs change.
