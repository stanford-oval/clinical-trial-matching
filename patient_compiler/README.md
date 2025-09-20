# patient_compiler/ -- Patient-Side Constraint Semantic Parsing

Implements the online patient-side semantic parsing pipeline described in Section 5 of the SatIR paper. Converts free-text patient clinical notes into canonical structured variables suitable for constraint matching.

> "Patient-side semantic parsing follows a simplified version of Stages 1-3 with more careful treatment of time-window qualifiers."

## Architecture

```
Patient Clinical Note (free-text narrative)
    |
    v
Stage 1: Patient State Extraction
    |  Rudimentary fact extraction
    |  Logical precision rewriting
    |  Entity surface expansion
    |  Verification at each sub-stage
    v
Stage 2: Entity Canonicalization
    |  Entity linking against SNOMED-CT
    |  Diagnosis inference (when symptoms described without explicit diagnosis)
    |  Non-canonical predicate identification
    v
Stage 3: Attribute Extraction
    |  Qualifier identification
    |  Attribute translation and verification
    v
Stage 4: Patient Coding
    |  Canonical variable generation with temporal constraints
    |  Demographics variable coding (age, sex)
    |  Diagnosis coding
    |  Time-window qualifiers (inthepast, inthefuture, now, foradurationof)
    v
Canonical Patient Representation
    (canonical.jsonl, demographics.jsonl, diagnosis.jsonl)
```

## Handling Patient Data Missingness (Section 5)

The patient compiler addresses two types of data missingness:

### Whole-Fact Missingness
When patient records lack information to support or refute specific constraints, the system applies the **salience principle**: any salient information about a patient's condition will be documented in their medical record. An LLM assesses whether absent information should be interpreted as supporting, refuting, or inconclusive.

### Under-Specificity
Medical records may be under-specified. A patient documented with "appendicitis" may match a trial for "Acute Appendicitis" (via subsumption <=_K), but should not match "Ruptured Suppurative Appendicitis" without explicit evidence. Salience of the targeted condition determines matching direction.

### Diagnosis Inference
When patient notes describe symptoms without explicit diagnosis, the system augments representation by inferring likely diagnoses:
- `PatientStateDifferentialDiagnoser` -- LLM-based differential diagnosis from symptoms
- `PatientStateExplicitDiagnoseExtractor` -- Extract explicitly stated diagnoses

## Modules

### modules/patient_state_extractor/ (Stage 1)
- `PatientStateRudimentaryExtractor` + Verifier -- Initial fact extraction from clinical notes
- `PatientStateLogicalPrecisionRewriter` + Verifier -- Make logical structure explicit
- `PatientStateEntitySurfaceExpander` + Verifier -- Expand entity span boundaries
- `PatientStateDifferentialDiagnoser` -- Infer diagnoses from symptom descriptions
- `PatientStateExplicitDiagnoseExtractor` -- Extract explicit diagnoses

### modules/entity_canonicalizer/ (Stage 2)
Patient-specific additions on top of `smt_core` base:
- `DiagnosisCanonicalizer` -- Specialized canonicalization for inferred diagnoses

### modules/attribute_extractor/ (Stage 3)
Patient-specific orchestrator composing shared attribute stages from `smt_core`.

### modules/patient_coder/ (Stage 4)
- `PatientCanonicalVariableCoder` -- Generate canonical variable names with temporal constraints
- `PatientDemographicsVariableCoder` -- Age, sex, pregnancy status
- `PatientDiagnosisCoder` -- Differential diagnosis variables
- `PatientDiagnoseClassifier` -- Classify diagnosis confidence
- `PatientCanonicalEntityEnricher` -- Enrich coded entities with ontology context
- `PatientCanonicalVariableOtherCandidatesCoder` -- Alternative candidate variables

## Usage

```bash
# Compile a single patient note
compile-patient <patient_id>

# Resume from checkpoint
compile-patient <patient_id> --resume-from canon

# Stop after a specific stage
compile-patient <patient_id> --stop-after program
```

## Output Artifacts

Per patient, outputs are written to `build/patient_coded_results/{patient_id}/{side}/`:

| File | Description |
|------|-------------|
| `canonical.jsonl` | Canonical facts with time windows (var_name, value, start/end hours, qualifiers) |
| `demographics.jsonl` | Age (years/months/days), sex |
| `diagnosis.jsonl` | Differential diagnoses with confidence levels |

Each fact carries temporal information:
- `start_time_in_hours`, `end_time_in_hours` -- Temporal interval
- `start_time_inclusive`, `end_time_inclusive` -- Endpoint inclusivity
- `kind` -- Timeframe category (inthepast, inthefuture, now, foradurationof)
- `is_root` -- Whether this is an original coded fact vs. derived
