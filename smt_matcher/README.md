# smt_matcher/ -- SMT-Based Eligibility Checking

Implements end-to-end SMT-based eligibility checking for individual patient-trial pairs. Given a compiled trial SMT program and a patient's structured data, uses the Z3 SMT solver to determine constraint satisfaction.

While `sql_retrieval/` provides fast, scalable retrieval across thousands of trials, `smt_matcher/` provides precise, per-pair eligibility determination with full constraint reasoning -- including non-canonical predicates, counting constraints, and complex logical structure that cannot be represented in the relational schema.

## Architecture

```
Trial SMT Program (build/ir/ or build/symtab/)
    +
Patient Data (clinical notes or structured variables)
    |
    v
Stage 1: SMTLeafCollector
    |  Extract leaf variables from SMT program
    |  Parse variable types, descriptions, enum values
    v
Stage 2: SMTVariableValueMiner
    |  LLM-based: mine variable values from patient data
    |  Maps each trial variable to patient evidence
    v
Stage 3: SMTProgramEvaluator
    |  Substitute mined values into SMT program
    |  Call Z3 solver for satisfiability
    |  Per-requirement and whole-program evaluation
    v
Result: {per_requirement: {R1: sat/unsat, ...}, overall: sat/unsat}
```

## Modules

### modules/smt_matcher/
- **`SMTMatcher.py`** -- Top-level orchestrator: chains leaf collection, value mining, and evaluation
- **stages/**:
  - `SMTLeafCollector.py` -- Z3 introspection to extract declared variables with types and inline JSON metadata
  - `SMTVariableValueMiner.py` -- LLM-based variable value extraction from patient notes (uses inclusion/exclusion-specific prompts)
  - `SMTProgramEvaluatorAllTogether.py` -- Evaluate entire SMT program at once via Z3
  - `SMTProgramEvaluatorPerCriterion.py` -- Evaluate per-requirement blocks individually (finer-grained eligibility reasoning)

## Input Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Symbol table | `build/symtab/{trial_id}_{side}_variable_index.json` | Variable registry with types and meanings |
| IR program | `build/ir/{trial_id}_{side}_program.smt2` | SMT-LIB program |
| Entity linkmap | `build/linkmap/{trial_id}_{side}_entities.json` | Entity-to-concept mappings |

## Usage

```bash
# Match a specific patient to a specific trial
match-trial --trial NCT03362970 --patient sigir-001

# With explicit paths
python -m smt_matcher.match_patient_to_trial \
    --trial NCT03362970 \
    --patient sigir-001 \
    --build-root ../build \
    --side both

# With ground-truth labels for evaluation
python -m smt_matcher.match_patient_to_trial \
    --trial NCT03362970 \
    --patient sigir-001 \
    --labels ../dataset/clinical_trial/splits/
```

## Output

JSON result per (trial, patient, side):
```json
{
  "trial_id": "NCT03362970",
  "patient_id": "sigir-001",
  "inclusion": {
    "per_requirement_grouped": {
      "R1": "sat",
      "R2": "unsat",
      "R3": "sat"
    },
    "overall_status_grouped": "unsat",
    "labels": {
      "inclusion_gpt4_eligibility": "...",
      "inclusion_expert_eligibility": "..."
    }
  },
  "exclusion": { ... }
}
```
