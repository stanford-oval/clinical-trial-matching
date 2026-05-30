# Build Artifacts Directory Structure

All components write to a shared `build/` directory, configurable via the `SATIR_BUILD` environment variable (default: `./build`).

```bash
export SATIR_BUILD=/path/to/build  # override for all components
```

## Directory Layout

```
build/
├── canon/                              # Canonical variable representations
│   ├── NCT*_inclusion_canonical_variables.json
│   └── NCT*_exclusion_canonical_variables.json
│
├── ir/                                 # Raw SMT-LIB intermediate representations
│   ├── NCT*_inclusion_program.smt2
│   └── NCT*_exclusion_program.smt2
│
├── ir_all_final/                       # Finalized IR (after repair/polarity/logic fixes)
│   ├── NCT*_inclusion_program.smt2
│   └── NCT*_exclusion_program.smt2
│
├── symtab/                             # Variable symbol tables (declare-const meanings)
│   ├── NCT*_inclusion_variable_index.json
│   └── NCT*_exclusion_variable_index.json
│
├── linkmap/                            # Entity-to-concept linkage maps
│   ├── NCT*_inclusion_entities.json
│   └── NCT*_exclusion_entities.json
│
├── requirements/                       # Extracted requirements (intermediate)
│   ├── NCT*_inclusion_requirements.json
│   └── NCT*_exclusion_requirements.json
│
├── disease/                            # Disease list processing outputs
│   └── NCT*_disease_link_filter_summary.json
│
├── default_vars/                       # Default variable values
│
├── positive_literals_categorized/      # Positive literal extraction
│   └── per_file/
│       └── NCT*_{side}_program.smt2.json
│
├── canon_projection/                   # CNF projections for SQL retrieval
│   ├── slice_ir/                       # Sliced IR components
│   ├── slice_ir_linked/                # With qualifier links
│   ├── slice_ir_linked_minified/       # Minified
│   └── slice_ir_linked_minified_projected/  # Final projected CNF
│
├── trial.db                            # Main SQLite database (trial-side)
│                                       # Tables: clauses, clause_literals, var_catalog,
│                                       #   trial_sides, trial_side_clauses, disease_list_items,
│                                       #   disease_accepted_alternatives,
│                                       #   positive_literal_list_items,
│                                       #   positive_literal_accepted_alternatives_expanded
│
└── patient_coded_results/              # Patient-side coded outputs
    └── {patient_id}/
        ├── inclusion/
        │   ├── canonical.jsonl         # Canonical facts with time windows
        │   ├── demographics.jsonl      # Age, sex
        │   └── diagnosis.jsonl         # Differential diagnoses
        └── exclusion/
            ├── canonical.jsonl
            ├── demographics.jsonl
            └── diagnosis.jsonl
```

## Which Component Writes Where

| Component | Writes To |
|-----------|-----------|
| `trial_compiler/compile_trial.py` | `canon/`, `ir/`, `symtab/`, `linkmap/`, `requirements/` |
| `trial_compiler/ir_finalizer/` | `ir_all_final/`, `symtab_final/` (reads from `ir/`) |
| `trial_compiler/disease_compiler/` | `disease/` |
| `patient_compiler/compile_patient.py` | `patient_coded_results/` |
| `db_indexer/trial_side/` | `trial.db`, `canon_projection/`, `positive_literals_categorized/` |
| `db_indexer/disease_side/` | `trial.db` (disease_list_items, disease_accepted_alternatives) |
| `db_indexer/patient_side/` | `trial.db` (facts_inclusion, facts_exclusion, patient_demographics) |
| `db_indexer/categorizer/` | `trial.db` (categorized variants of disease/poslit tables) |
| `db_indexer/sibling_side/` | `trial.db` (sibling alternative tables) |
| `smt_matcher/` | Reads from `symtab/`, `ir/`, `linkmap/` |
| `sql_retrieval/ops/` | Reads from `trial.db` |

## Environment Variables

| Variable | Default | Used By |
|----------|---------|---------|
| `SATIR_BUILD` | `./build` | All components |
| `OPENAI_ENDPOINT` | (required) | All LLM-calling components |
| `OPENAI_API_KEY` | (required) | All LLM-calling components |
| `OPENAI_MODEL` | auto-detect from endpoint | All LLM-calling components |
| `SNOWSTORM_BASE` | `http://localhost:8080` | Entity canonicalization, ontology lifting |
| `TRIAL_DATA` | `../dataset/clinical_trial` | Data loading |

## Checkpoint Directories

Each compiler maintains its own checkpoint directory for resumability:

```
trial_compiler/checkpoints/        # Trial compilation checkpoints
patient_compiler/checkpoints/      # Patient compilation checkpoints
```

Checkpoints are keyed by `{trial_or_patient_id}_{stage}.chkpt.json` and enable `--resume-from` / `--stop-after` flags.
