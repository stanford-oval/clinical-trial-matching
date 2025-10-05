# trial_compiler/ -- Trial-Side Constraint Semantic Parsing

Implements the offline trial-side semantic parsing pipeline described in Section 2 of the SatIR paper. Converts free-text clinical trial eligibility criteria into formal SMT-LIB constraints.

> "We use Large Language Models (LLMs) to convert informal reasoning -- regarding ambiguity, implicit clinical assumptions, and incomplete patient records -- into explicit, precise, controllable, and interpretable formal constraints."

## Architecture

The trial compiler implements the paper's three-stage semantic parsing pipeline:

```
Trial Document (inclusion/exclusion criteria)
    |
    v
Stage 1: Structured Decomposition of Requirements
    |  Polarity reversal (exclusion -> unified form)
    |  Span contiguity (entity phrases made contiguous)
    |  Logical structure (AND/OR/NOT made explicit)
    |  Decomposition (into modular, self-contained units)
    v
Stage 2: Entity Canonicalization
    |  Entity linking against SNOMED-CT ontology
    |  Non-canonical predicate identification
    v
Stage 3: Incremental SMT Programming
    |  Fragment-by-fragment translation to SMT-LIB
    |  Validation: parsing, solver, and semantic checks
    v
SMT-LIB Program (~360 lines per trial)
    |
    v
IR Finalization (ir_finalizer/)
    |  Polarity fixes, logic fixes, repair
    |  Underconstraint tightening, meaning enrichment
    |  Strict guardrail validation
    v
Final IR + Symbol Table
```

## Sub-Components

### compile_trial.py (107 .py files, 101 prompts)
Main orchestrator (`TrialPipeline` class) with stages:
1. **PREPROC** -- `RequirementContextPreprocessor`: cohort splitting, context normalization
2. **EXTRACT** -- `RequirementExtractor`: structured decomposition (5 sub-stages)
3. **CANON** -- `EntityCanonicalizer`: entity linking (from smt_core)
4. **ATTR** -- `AttributeExtractor`: qualifier and attribute extraction
5. **PROGRAM** -- `SMTProgrammer`: incremental SMT-LIB generation with validation

Each stage is checkpointed and resumable via `--resume-from` / `--stop-after`.

### modules/requirement_extractor/ (Stage 1)
Implements structured decomposition with these sub-stages:
- `RequirementContextPreprocessor` -- Normalizes trial context, splits multi-cohort trials
- `RequirementRudimentaryExtractor` + Verifier -- Initial requirement extraction
- `RequirementEntitySurfaceExpander` + Verifier -- Expands entity span boundaries
- `RequirementLogicalPrecisionRewriter` + Verifier -- Makes AND/OR/NOT explicit
- `RequirementDecomposer` + Verifier -- Breaks into atomic, self-contained constraints
- `RequirementHardSoftClassifier` -- Labels constraints as hard or soft
- `RequirementContradictCriterionRewriter` -- Handles contradictions between inc/exc

### modules/smt_programmer/ (Stage 3)
Incremental SMT programming with two strategies:
- **SMTIncrementalProgrammer/** -- Per-requirement incremental translation:
  - `SMTIncrementalReusableVariableIdentifier` -- Identifies variable reuse
  - `SMTIncrementalDemographicsVariableNamer` -- Age/sex/pregnancy variables
  - `SMTIncrementalCanonicalVariableNamer` -- Canonical entity variables
  - `SMTIncrementalFreeVariableNamer` -- Non-canonical entity variables
  - `SMTIncrementalTranslator` -- Natural language to SMT-LIB
  - `SMTIncrementalSolverBasedValidator` -- Z3 parsing + consistency check
  - `SMTIncrementalVerifier` -- LLM semantic faithfulness check
- **SMTOnepassProgrammer/** -- Full-program translation (alternative strategy)
- **RegistryUtils/** -- Variable registry codec for tracking declared variables

### ir_finalizer/ (28 .py files, 8 prompts)
Post-compilation IR repair and validation pipeline (from `fsrc/`):
- `smt_repair_module.py` -- General syntax/semantic repair
- `smt_polarity_fix_module.py` -- Exclusion polarity correction
- `smt_logic_fix_module.py` -- Logical encoding fixes
- `smt_fix_underconstraint_module.py` -- Tighten underencoded constraints
- `smt_variable_meaning_enricher_module.py` -- Enrich declare-const annotations
- `smt_criteria_gate_module.py` -- LLM gate: verify criteria substantively present
- `orchestrate_ir_fixes.py` -- Fixed-workflow orchestrator with strict compliance
- `orchestrate_ir_fixes_simplified.py` -- Parallel threadpool orchestrator
- `build_symtab_final.py` -- Extract variable symbol table from final IR

### disease_compiler/ (18 .py files, 16 prompts)
Disease list processing pipeline (from `dsrc/`), described in Section 5.2 appendix:

```
Extraction --> Elimination --> Preprocessing --> Canonicalization
```

- `DiseaseListExtractor` -- LLM-based high-recall disease extraction from trial text
- `DiseaseListEliminator` -- Relevance-based filtering of non-target diseases
- `DiseaseListPreprocessor` -- Conservative string normalization (whitespace, aliases, dedup)
- `DiseaseListRevisor` -- Deterministic + optional LLM revision
- `DiseaseLogicCapturer` -- Extract Boolean logic (AND/OR/NOT) across diseases
- `DiseaseCanonicalizer` -- Map to SNOMED-CT via vector search + LLM filtering

## Usage

```bash
# Compile a single trial (both inclusion and exclusion)
compile-trial NCT03362970 --side both

# Resume from a checkpoint
compile-trial NCT03362970 --side inclusion --resume-from canon

# Stop after a specific stage
compile-trial NCT03362970 --side both --stop-after program

# Use canonical subcohort data
compile-trial NCT03362970 --preproc-source canonical --canonical-subcohort-dir ../canonical_subcohort_results/

# Compile disease list for a trial
python -m trial_compiler.disease_compiler.compile_disease NCT03362970
```

## Output Artifacts

| Stage | Output Path | Format |
|-------|-------------|--------|
| PREPROC | `checkpoints/preproc/` | JSON checkpoint |
| CANON | `build/canon/{trial_id}_{side}_canonical_variables.json` | Canonical variable mappings |
| PROGRAM | `build/ir/{trial_id}_{side}_program.smt2` | SMT-LIB program |
| FINAL | `build/ir_all_final/{trial_id}_{side}_program.smt2` | Validated final IR |
| SYMTAB | `build/symtab_final/{trial_id}_{side}_variable_index.json` | Variable symbol table |
| DISEASE | `build/disease/{trial_id}_disease_link_filter_summary.json` | Canonical disease mappings |
