# sql_retrieval/ -- Objective-Conditioned SQL Retrieval

Implements the scalable retrieval engine described in Section 4.3 of the SatIR paper. Uses SQL queries over the relational schema to match patients to trials in 2.95 seconds per patient across 3,621 trials.

> "Matches for a given objective theta are retrieved with a straightforward SQL query over the database tables: SQL(O, CD, PD, theta)."

## Architecture

The retrieval query operates in three levels:

```
Level 1: Atom-Level Matching
    Join trial and patient atom tables on:
    - Same canonical predicate pi
    - Retained qualifier fields
    - Compatible comparison conditions
    --> Produces compatible atom pairs

Level 2: Clause-Level Aggregation
    Join compatible atom pairs with DA table
    Clause marked "supported" if >= 1 atom matches
    --> Produces per-clause support decisions

Level 3: CNF/Entity-Level Resolution
    Aggregate supported clauses back to trials
    Return (trial, patient) only if ALL
    retrieval-relevant clauses are supported
    --> Final retrieval set
```

The retrieval objective theta determines which trial-side atoms are relevant and which patient-side atoms may support them, restricting allowable SQL joins.

## Three Retrieval Objectives

| Mode | Description | What Matches |
|------|-------------|--------------|
| **ModeCCR** | Treat chief complaint | Trials treating patient's chief complaint |
| **ModeAll** | Treat any condition | Trials treating any patient condition |
| **ModeAllExplore** | Relevant to any condition | Trials relevant to any condition (broadest) |

## Sub-Components

### ops/ (28 .py files)
Core SQL retrieval composition and trial-patient kit building:
- **`compose_trial_eval.py`** -- Main evaluation driver: composes retrieval queries across all objectives
- **`trial_eval_primitives.py`** -- Core matching functions:
  - `inclusion_gap()` -- Find patients meeting inclusion criteria
  - `disease_hits()` -- Disease-matching patients via `disease_accepted_alternatives`
  - `positive_literal_hits()` -- Positive-literal matching via time-interval overlap
  - `prevention_hits()` -- Prevention-scoped matching
  - `check_constraints()` -- SMT constraint solving via Z3
- **`build_patient_trial_kits.py`** -- Assemble per-patient/trial kits with rankings and explanations
- **`llm_judge.py`** -- LLM-based final matching for residual non-formal constraints

### eval/ (45 .py files)
Quantitative evaluation against benchmarks:
- **`compare_smt_vs_trialgpt_avgcutoff.py`** -- Head-to-head SatIR vs TrialGPT comparison
- **`compare_satir_vs_trialgpt5_avgcutoff.py`** -- Updated comparison with GPT-5
- **`eval_pr_rec_at_k*.py`** -- Precision/recall at various cutoffs
- **`find_smt_false_positives.py`** -- False positive analysis
- **`count_trialgpttop200_fn.py`** -- False negative discovery in TrialGPT top-200
- **`run_smt_retrieval_eval.py`** -- End-to-end SMT retrieval evaluation
- **`run_trialgpt_ref_eval.py`** -- TrialGPT reference evaluation
- **`judge_eligibility.py`** / **`judge_relevance.py`** -- LLM-based relevance/eligibility judges
- **`build_clinician_review_workbook.py`** -- Generate clinician review spreadsheets
- **`cache_utils.py`** -- Shared pair caching for evaluation

### meval/ (5 .py files)
LLM-based adjudication between systems:
- **`adjudicate_smt_vs_nl.py`** -- Three-system adjudication (SatIR vs NL baseline)
- **`destyle_three_systems_single_case.py`** -- Case-level de-styled comparison

## Key Results

| Metric | SatIR | TrialGPT | Improvement |
|--------|-------|----------|-------------|
| Treat-chief (avg trials/patient) | 3.25 | 2.17 | +50% |
| Treat-any (avg trials/patient) | 5.12 | 2.98 | +72% |
| Relevant-to-any (avg trials/patient) | 11.76 | 8.93 | +32% |
| Treat-chief recall | 93.66% | 62.44% | +31 pts |
| Treat-any recall | 92.07% | 53.66% | +38 pts |
| Relevant-to-any recall | 92.53% | 70.27% | +22 pts |
| Query time (3,621 trials) | **2.95s** | -- | -- |

Patient-level: SatIR wins on 20-35 out of 59 patients depending on objective, while serving more patients with at least one useful trial (54 vs 50 for relevant-to-any).

## Usage

```bash
# Run full retrieval evaluation
python -m sql_retrieval.ops.compose_trial_eval \
    --db /path/to/trial.db \
    --mode all

# Run comparison against TrialGPT
python -m sql_retrieval.eval.compare_smt_vs_trialgpt_avgcutoff

# Run LLM-based adjudication
python -m sql_retrieval.meval.adjudicate_smt_vs_nl
```
