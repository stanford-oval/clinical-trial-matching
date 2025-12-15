# Typed missingness policy — version 5

Policy version: `v5`.

Restructured taxonomy grounded in established clinical-data standards:

- Disease/Disorder: 2x2 (common/uncommon x chronic/acute) + Other
- Symptom/Sign: temporal-scope-driven (past, current, exam current/past, other)
- Numerical: raw values collapse to one row; thresholds route by semantic
- Procedure: SNOMED CT-style 6-type taxonomy (imaging, diagnostic, surgical,
  therapeutic non-surgical, vaccination, routine) plus planned/scheduled
- Demographic, Consent/Setting, Other: simple

## States (unchanged)

| State | Meaning |
|---|---|
| `explicit_true` | Chart documents the predicate holds. |
| `explicit_false` | Chart documents the predicate does NOT hold. |
| `assumed_false` | Chart silent; policy treats as False (informative absence). |
| `assumed_true` | Chart silent; policy treats as True (rare). |
| `unknown_defer` | Chart silent; policy defers to the in-person visit. |

## Inversion constraint

For predicate naming form `has_X` (positive polarity) vs `has_no_X`
(negative polarity), the silence defaults must be inverses:

- `assumed_false` <-> `assumed_true`
- `unknown_defer` <-> `unknown_defer` (self-inverse)

Every row in v5 satisfies this constraint.

## Bayesian justification for common/uncommon split

The silence rule on a category is determined by the posterior probability
the condition is present given chart silence:

```
P(present | silent) proportional to P(silent | present) * P(present)
```

- **Uncommon conditions** (low base rate) -> silence is strong evidence of
  absence -> `assumed_false / assumed_true`
- **Common conditions** (high base rate) -> silence is weak evidence
  either way -> `unknown_defer`

This drives both the Disease/Disorder 2x2 and the Numerical Threshold
semantic split.

## Procedure documentation-rigor justification

| Type | Documentation pattern | Silence rule |
|---|---|---|
| Imaging | Radiology reports archived per study, but specific reports may not be in the chart excerpt | defer |
| Diagnostic | Procedure notes archived, same caveat as imaging | defer |
| Surgical | Lifelong on surgical history list | assumed |
| Therapeutic non-surgical | Treatment courses heavily documented in active-treatment list | assumed |
| Vaccination | Separate registries, often incomplete in main chart | defer |
| Routine | No discrete chart entry by definition | defer |
| Planned/Scheduled | Future events not chart-retrievable | defer |

## Numerical Threshold semantic routing

Raw numerical predicates always default to `unknown_defer`. Thresholded
predicates (numeric comparisons that encode a clinical concept) are
classified by what the threshold encodes:

| Encodes | Silence rule |
|---|---|
| Uncommon disease/disorder/symptom (e.g. `creatinine > 2`, `platelet < 100`) | `assumed_false / assumed_true` |
| Common disease/disorder/symptom (e.g. `bp_sys > 140`, `bmi > 30`) | `unknown_defer` |
| Trial-arbitrary cutoff with no named clinical-concept interpretation | `unknown_defer` |

## Aggregation rule (unchanged from v3)

A predicate is *load-bearing-failing* iff:

| Demand x State | Failing? |
|---|---|
| positive demand + explicit_false or assumed_false | YES |
| negative demand + explicit_true or assumed_true | YES |
| any demand + unknown_defer | NO |
| alternatives / value_constrained (any state) | NO (SMT solver decides) |

Final verdict:
- `ineligible` iff at least one load-bearing-failing predicate exists.
- Otherwise `eligible`.
