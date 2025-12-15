"""Typed missingness policy — clinical-trial eligibility matching system.

A two-track design that lets a deterministic SMT-based aggregator be
compared directly against an LLM matcher under a shared typed
missingness policy.

Modules:
    policy         silence-rule lookup keyed by (Level1, Level2) categories
    miner          Stage 1 — strict-extractive chart value mining
    classifier     Stage 2 — typed predicate -> (Level1, Level2) category
    smt_track      Stage 3 — Z3-based deterministic aggregation
    nl_track       NL eligibility matcher with V1-V4 prompt ablation
    zspm_track     Stanford-Koopman-style matcher with V1-V4 ablation

Assets:
    prompts/       all prompt templates (active + backup/)
    policies/      typed_missingness.csv + .md (active + backup/)
"""
from . import policy
from . import miner
from . import classifier
from . import smt_track
from . import threshold
# nl_track and zspm_track modules are TODO — wire when needed

__all__ = ['policy', 'miner', 'classifier', 'smt_track', 'threshold']
