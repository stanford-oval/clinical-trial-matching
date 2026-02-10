# matchers

> Side-by-side reference implementations of every system compared in *Cross-Validated Symbolic and Natural-Language Reasoning for Auditable Clinical-Trial Prescreen* (EMNLP submission).

A single Python package that exposes nine clinical-trial eligibility systems — from external baselines to fully-auditable SMT-based pipelines — through a common interface so you can inspect, compare, and run them on any (patient, trial) pair in the SIGIR clinical-trial dataset.

```python
from matchers import variants

d = variants.smt_lm_evidence_arbiter("sigir-201414__NCT02192320")
print(d.decision)        # "eligible" / "ineligible"
print(d.is_auditable())  # True / False
for step in d.audit_trail:
    print(step.stage, "->", step.decision, ":", step.rationale[:80])
```

## Why this exists

The paper compares nine systems across multiple metrics (Acc, F1, F2, Cohen's κ, auditability). Several reviewers will ask: *"Did you compare to a stronger NL baseline?"* — yes, and you can run any of them yourself in one line. The paper's tables are reproducible from the cached LM outputs in this repository's experiment directories.

## The nine systems

| Variant | Group | Final decider | Auditable? | Headline F1 / F2 / κ |
|---|---|---|---|---|
| `trialgpt` | external baseline (Yang et al. 2023) | LLM (sentence-level) | no | 0.438 / 0.344 / 0.252 |
| `lm_only` | our ablation (basic prompt) | LLM | no | 0.755 / 0.674 / 0.614 |
| `lm_only_prescreen` | stronger NL-only baseline | LLM | no | 0.799 / 0.886 / 0.619 |
| `multiagent_nl` | strongest NL-only baseline (4 LM stages) | LM-deciding from LM-audit | no | 0.754 / 0.800 / 0.510 |
| `smt_raw` | our ablation (no review) | SMT solver | **yes** | 0.786 / 0.763 / 0.621 |
| `smt_atoms_arbiter` | our SMT-based pipeline | SMT solver | **yes** | 0.790 / 0.838 / 0.581 |
| **`smt_lm_evidence_arbiter`** | **recommended (auditable)** | SMT solver | **yes** | **0.804** / 0.835 / 0.621 |
| `hybrid_loose` | our hybrid (max recall) | accept-if-either | partial | 0.811 / 0.875 / 0.613 |
| **`hybrid_strict`** | **recommended (max F1)** | accept-if-either | partial | **0.824** / 0.869 / **0.652** |

(All numbers are 5-judge mean on the 538 parsed pairs; see paper Table 1.)

## Quickstart

```bash
# Inspect all 9 variants on a single pair, with full audit trails
python -m matchers.inspect_one sigir-20141__NCT02001545

# Just one variant
python -m matchers.inspect_one sigir-20141__NCT02001545 \
    --variant smt_lm_evidence_arbiter

# Aggregate accuracy across the 122-pair disagreement subset
python -m matchers.inspect_one --all-disagreements --summary
```

## A worked example: side-by-side comparison

For `sigir-20141__NCT02001545` (a "STEMI/NSTEMI not in chart" case where gold = ineligible):

```
trialgpt              ✓ ineligible  AUDITABLE
  trialgpt_score: ineligible

lm_only               ✓ ineligible  AUDITABLE
  lm_judge: ineligible — "the patient presented with chest pain but
            no documented STEMI/NSTEMI diagnosis"

smt_raw               ✗ eligible    AUDITABLE
  atom_mining: 10 atoms
  smt_solve: eligible — solver found a satisfying assignment
            (no blocking atoms; chart silence let the program be sat)

hybrid_strict         ✗ eligible    AUDITABLE
  smt_solve: eligible
  lm_judge: ineligible
  accept_if_either: eligible (authority=smt_solver)
```

This case is one of 56 in the paper where the SMT-miner's silence-default
behavior over-asserted; the LM-judge correctly caught it. See paper §6.7.

## Architecture

Every variant returns a `Decision`:

```python
@dataclass
class Decision:
    pair_id: str
    variant: str
    decision: str                 # "eligible" / "ineligible"
    reasoning: str                # one-sentence summary
    audit_trail: List[AuditStep]  # ordered list of the system's reasoning steps

    def is_auditable(self) -> bool:
        """True iff every step traces to chart-grounded atoms.
        False only when accept_if_either rests on LM-judge alone."""
```

Every step is an `AuditStep`:

```python
@dataclass
class AuditStep:
    stage: str              # "atom_mining", "smt_solve", "lm_judge",
                            # "atoms_only_arbiter", "lm_evidence_arbiter",
                            # "accept_if_either", "extractor", "critic", ...
    decision: str = None    # this stage's verdict, if any
    rationale: str = ""     # human-readable
    evidence: dict = ...    # structured data
```

The same `stage` names are used across all nine variants. Reading any two
variants' `audit_trail` on the same pair shows exactly where they diverge.

## How they differ structurally

```
                                  LM-only paths              SMT-based paths
                                  ─────────────              ───────────────
chart + criteria  ─┬─→ trialgpt                   ─→ outputs verdict
                   │
                   ├─→ lm_only                    ─→ outputs verdict
                   │   (one LLM call, basic prompt)
                   │
                   ├─→ lm_only_prescreen          ─→ outputs verdict
                   │   (one LLM call, prescreen-doctrine prompt)
                   │
                   ├─→ multiagent_nl              ─→ outputs verdict
                   │   (extractor → critic → arbiter → decider, all LM)
                   │
                   ├──────────────→ smt_raw                              ─→ SMT solver decides
                   │                (atom miner → Z3)
                   │
                   ├──────────────→ smt_atoms_arbiter                    ─→ SMT solver decides
                   │                (smt_raw + LM auditor on rejection,
                   │                 auditor sees atoms only)
                   │
                   ├──────────────→ smt_lm_evidence_arbiter      ◀ recommended (auditable)
                   │                (smt_raw + LM auditor seeing
                   │                 LM-judge rationale as evidence)
                   │
                   ├──────────────→ hybrid_loose       ◀ recommended for max recall
                   │                (smt_raw + parallel LM-judge,
                   │                 accept-if-either,
                   │                 atoms-only arbiter on both-reject)
                   │
                   └──────────────→ hybrid_strict      ◀ recommended for max F1
                                    (smt_raw + parallel LM-judge,
                                     accept-if-either,
                                     LM-evidence arbiter on both-reject)
```

## What's in this package

| File | What's in it |
|---|---|
| `__init__.py` | Imports the 9 variants for one-line use |
| `schema.py` | `Decision` and `AuditStep` dataclasses (shared types) |
| `data.py` | Per-pair loader (chart, criteria, atoms, LM-judge output, gold) |
| `prompts.py` | All 5 prompts in one place: atoms-only arbiter, LM-evidence arbiter, basic LM-judge, prescreen-doctrine LM-judge, plus the 3 multi-agent prompts |
| `variants.py` | The 9 variants — each a self-contained function |
| `inspect_one.py` | CLI |

## Reproducibility

The 9 variants pull cached LM outputs from sibling `experiments/` directories.
Re-running on a new pair without a cache hit returns the SMT decision unchanged
with an `(uncached)` audit step; populate the cache by running the cache-producing
scripts (`experiments/97_smt_based/run_smt_based.py`,
`experiments/99_counterfactual_lm/run_better_nl_full.py`,
`experiments/100_multiagent_nl/run_multiagent.py`).

The cached numbers exactly reproduce the paper's tables. To verify:

```bash
python -m matchers.inspect_one --all-disagreements --summary
```

## Citing

```bibtex
@inproceedings{xxx2026smt-trial-matching,
  title  = {Cross-Validated Symbolic and Natural-Language Reasoning for Auditable Clinical-Trial Prescreen},
  author = {Anonymous},
  booktitle = {EMNLP},
  year   = {2026}
}
```

## License

MIT (see `LICENSE`).
