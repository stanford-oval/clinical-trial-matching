"""Counterfactual modifier package.

Routes a RationaleSpec (from rationale_generators) to the correct modifier:
  - atom_target_modifier:  for kind='atom_targets' (aegis)
  - rationale_modifier:    for kind='rationale_text' (everything else)

Both modifiers produce a counterfactual chart (cf_chart) and call gpt-5 with
shared coherence/minimality rules. The validator (simclin) is invoked
separately by the experiment driver.
"""
from __future__ import annotations

from .atom_target_modifier import generate_cf_gpt5_targets
from .rationale_modifier   import generate_cf_gpt5_rationale


def apply(rationale_spec: dict, original_chart: str, trial_text: str) -> dict:
    """Apply the right modifier given a RationaleSpec.

    Returns: {cf_chart, modifier, source_rationale_kind, source_rationale}
    """
    kind = rationale_spec.get('kind')
    if kind == 'atom_targets':
        cf = generate_cf_gpt5_targets(
            original_chart,
            rationale_spec.get('atom_targets') or [],
            preserve   = rationale_spec.get('preserve') or [],
            other_atoms= rationale_spec.get('other_atoms') or [],
        )
        return {
            'cf_chart':              cf,
            'modifier':              'gpt-5.atom_target',
            'source_rationale_kind': 'atom_targets',
            'source_rationale':      rationale_spec.get('atom_targets'),
            'source':                rationale_spec.get('source'),
        }
    elif kind == 'rationale_text':
        cf = generate_cf_gpt5_rationale(
            original_chart, trial_text,
            rationale_spec.get('rationale','') or '',
            rationale_spec.get('supports','') or '',
        )
        return {
            'cf_chart':              cf,
            'modifier':              'gpt-5.rationale_text',
            'source_rationale_kind': 'rationale_text',
            'source_rationale':      rationale_spec.get('rationale'),
            'source_supports':       rationale_spec.get('supports'),
            'source':                rationale_spec.get('source'),
        }
    raise ValueError(f"unknown rationale_spec kind: {kind!r}")
