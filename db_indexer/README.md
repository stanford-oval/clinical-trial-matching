# db_indexer/ -- Database Indexing

Implements the database construction and indexing pipeline described in Section 4 of the SatIR paper. Populates the relational schema that enables fast, scalable SQL-based retrieval.

> "SatIR uses two databases: a patient database (PD) and a clinical trial database (CD). Each contains five tables."

## Relational Schema (Section 4)

SatIR stores constraints in five relational tables per entity type:

```
ECNF(e, cnf)          Entity-to-CNF mapping
CNFD(cnf, d)           CNF-to-clause mapping (conjunction of disjunctions)
DA(d, a)               Clause-to-atom mapping (disjunction of atoms)
AB(a, pi, cmp, t)      Boolean atom table (uniquified)
AN(a, pi, cmp, t)      Numerical atom table with bounds and endpoint inclusivity
```

Each trial constraint TC(c) is converted from an SMT formula to quantifier-free Conjunctive Normal Form:

    CNF(e) = AND(i=1..m) d_i,  where  d_i = OR(j=1..n_i) a_ij

Quantifier elimination removes variables and subformulas involving non-canonical predicates, producing a conservative (recall-preserving) projection.

## Sub-Components

### trial_side/ (30 .py files)
Core trial-side indexing pipeline, orchestrated by `runall_trialside.py`:

| Step | Script | Description |
|------|--------|-------------|
| normalize | `normalize_units.py` | Normalize measurement units in SMT |
| slice | `smt_ir_slicer.py` | Slice constraint IR into components |
| link | `smt_qualifier_linker.py` | Link SNOMED qualifier relations |
| minify | `batch_minify_canon.py` | Minimize canonical representations |
| project | `smt_projector.py` | Project SMT to CNF (conservative) |
| db | `smt_clause_db.py` | Build clause database (ECNF/CNFD/DA/AB/AN) |
| find_poslit | `find_positive_canon_literals.py` | Extract positive literals from SMT logic |
| poslit | `ingest_positive_literals.py` | Ingest positive literals to DB |
| decide | `hop_policy_decider_lineage_mgsr.py` | LLM-based accept/reject with lineage tracking |
| lift | `ontology_lifter.py` | SNOMED ancestor lifting for positive literals |

**Positive Literal Extraction (Section 4.2 / Appendix A.3):**
Recovers canonical clinical propositions that are positively enforced by eligibility criteria. Uses polarity-aware traversal of the SMT program, inline shallow Boolean helper definitions, and lightweight implication-based expansion. More semantically grounded and broader than disease-list metadata alone.

### patient_side/ (37 .py files)
Multi-pass patient fact enrichment pipeline, orchestrated by `run_multi_pass_pipeline.py`:

**7 Enrichment Stages (per pass):**
1. **ISA** (`enrich_with_isa.py`) -- Lift findings/procedures/observables via SNOMED hierarchy (<=_K)
2. **IMP** (`apply_schema_implications.py`) -- Apply schema implications with timeframe handling
3. **F2P** (`enrich_findings_to_procedure.py`) -- Map findings to procedures via SNOMED interprets relation
4. **F2OE** (`enrich_findings_to_observable_entity.py`) -- Map findings to observable entities
5. **F2P_REL** (`enrich_findings_to_procedure_via_other_relations.py`) -- Extended procedure mapping
6. **P2F** (`enrich_procedure_to_finding.py`) -- Derive findings from procedures
7. **OE2F** (`enrich_observable_entity_to_finding.py`) -- Derive findings from observables

Enrichment encodes subsumption relations <=_K, <=_R, <=_causal, <=_Q as additional constraints injected into the patient fact database.

**Ingestion** (`ingest_patient_facts_to_db.py`): Populates `facts_inclusion`, `facts_exclusion`, `patient_demographics` tables with time-aware facts.

### disease_side/ (8 .py files)
Disease list ingestion + SNOMED ancestor lifting:
- `trial_disease_to_sqlite.py` -- Ingest disease JSON into `disease_list_items` table
- `disease_ontology_lifter.py` -- Build ancestor hierarchies, populate `disease_accepted_alternatives`

### disease_categorized_side/ (8 .py files)
Categorized disease pipeline (act variant) -- same structure as disease_side with categorization-aware ingestion.

### disease_categorized_nonact_side/ (9 .py files)
Categorized disease pipeline (nonact variant) -- for non-active/historical trial conditions.

### categorizer/ (12 .py files)
Disease and positive literal categorization into act/nonact/prevent:
- `categorize_disease_and_positive_literals.py` -- Main orchestrator
- `modules/disease_categorizer.py` -- Disease intent categorization (treat/prevent/diagnose)
- `modules/positive_literal_categorizer.py` -- Positive literal categorization
- `modules/joint_specificity_filter.py` -- Cross-entity specificity filtering

### sibling_side/ (7 .py files)
Sibling overlap detection and lifting -- finds semantically related but non-hierarchical concept alternatives:
- `ingest_sibling_alternatives.py` -- Ingest sibling relationships
- `ontology_lifter.py` / `ontology_lifter_clause.py` -- Lift via SNOMED
- `sibling_lift_materializer.py` -- Materialize lifted siblings
- `sibling_positive_lift_materializer.py` -- Materialize for positive literals

### control_group_identifier/ (3 .py files)
Identifies control group cohorts within multi-arm trials to avoid matching patients to placebo/control arms.

### default_value_side/ (2 .py files)
Generates and ingests default variable values for commonly referenced clinical variables.

### patient_geography_side/ (2 .py files)
Ingests patient geographic information for geography-constrained trial matching.

### rules/
Static configuration:
- `schema_implications.json` -- Implication rules for fact enrichment
- `mapping/*.csv` -- SNOMED relation mappings (finding-to-procedure, etc.)

## Database Tables Summary

### Trial-Side Tables
| Table | Description |
|-------|-------------|
| `clauses` | Clause catalog |
| `clause_literals` | Clause member atoms |
| `numerical_clauses` | Numerical constraint clauses |
| `numerical_predicates` | Numerical predicate definitions |
| `var_catalog` | Variable registry |
| `trial_sides` | Projected trial files |
| `trial_side_clauses` | Trial-to-clause mapping |
| `positive_literal_list_items` | Extracted positive literals |
| `positive_literal_accepted_alternatives` | Accepted alternatives with ancestors |
| `disease_list_items` | Base diseases per trial |
| `disease_accepted_alternatives` | Disease alternatives (self + kept ancestors) |

### Patient-Side Tables
| Table | Description |
|-------|-------------|
| `facts_inclusion` | Inclusion-side facts with timeframes |
| `facts_exclusion` | Exclusion-side facts with timeframes |
| `patient_demographics` | Age, sex per patient |
| `vw_fact_timeframes` | Union view with timeframe processing |

## Usage

```bash
# Run full trial-side indexing pipeline
index-db trial --build /path/to/build

# Run patient-side enrichment
index-db patient

# Or run individual steps
python -m db_indexer.trial_side.runall_trialside --steps find_poslit poslit decide lift
python -m db_indexer.disease_side.build_disease_pipeline
python -m db_indexer.patient_side.run_multi_pass_pipeline
```
