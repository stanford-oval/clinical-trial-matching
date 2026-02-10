"""
matchers — the seven systems compared in the EMNLP submission.

Each variant is exposed as a function with a stable signature:
    decide(pair_id: str) -> Decision

where Decision contains:
    - decision: "eligible" | "ineligible"
    - audit_trail: list of structured records describing every step
    - reasoning: human-readable summary

Variants (see VARIANTS dict for the full mapping to paper Table 1):

    External baseline:
        trialgpt                        — sentence-level LLM scorer (Yang et al. 2023)

    Single-component ablations (ours):
        lm_only                         — single LLM call reads chart + criteria
        smt_raw                         — atom miner + Z3 only, no review

    SMT-based pipelines (ours; fully auditable; SMT solver decides everything):
        smt_atoms_arbiter               — SMT-raw + LLM auditor on rejects (atoms only)
        smt_lm_evidence_arbiter         — SMT-raw + LLM auditor + LM-judge rationale as evidence
                                          (the auditable variant we recommend)

    Hybrid pipelines (ours; partial audit; LM-judge has direct accept-authority):
        hybrid_loose                    — accept-if-either + atoms-only arbiter on both-reject
        hybrid_strict                   — accept-if-either + LM-evidence arbiter on both-reject
                                          (the hybrid variant we recommend for max F1)

Use:
    from matchers import variants
    d = variants.smt_lm_evidence_arbiter("sigir-20141__NCT00000402")
    print(d.decision, d.reasoning)
    for step in d.audit_trail:
        print(step)
"""
from matchers.variants import (
    trialgpt,
    lm_only,
    lm_only_prescreen,
    multiagent_nl,
    smt_raw,
    smt_atoms_arbiter,
    smt_lm_evidence_arbiter,
    hybrid_loose,
    hybrid_strict,
    VARIANTS,
    VARIANT_GROUPS,
)
from matchers.schema import Decision, AuditStep

__version__ = "0.1.0"
__all__ = [
    "trialgpt", "lm_only", "lm_only_prescreen", "multiagent_nl",
    "smt_raw", "smt_atoms_arbiter", "smt_lm_evidence_arbiter",
    "hybrid_loose", "hybrid_strict",
    "VARIANTS", "VARIANT_GROUPS",
    "Decision", "AuditStep",
]
