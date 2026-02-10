"""TrialGPT rationale generator: filter to blocker rows.

TrialGPT tags each per-criterion verdict with one of:
  - inclusion side: [included] | [not included] | [not enough information]
  - exclusion side: [excluded] | [not excluded] | [not enough information]

Blockers:
  - inclusion [not included] = patient fails inclusion → blocker
  - exclusion [excluded]      = exclusion fires      → blocker

Drops [included]/[not excluded] rows (passing-criterion facts the patient
already meets) so the modifier only sees actionable blockers.
"""
from __future__ import annotations


_INC_BLOCKER_TAGS = ('[not included]',)
_EXC_BLOCKER_TAGS = ('[excluded]',)
_INC_PASS_TAGS    = ('[included]',)
_EXC_PASS_TAGS    = ('[not excluded]',)
_UNCERTAIN_TAGS   = ('[not enough information]',)


def _classify(line: str, section: str):
    blockers = _INC_BLOCKER_TAGS if section == 'inc' else _EXC_BLOCKER_TAGS
    passing  = _INC_PASS_TAGS    if section == 'inc' else _EXC_PASS_TAGS
    for tag in blockers:
        if tag in line: return 'blocker'
    for tag in passing:
        if tag in line: return 'passing'
    for tag in _UNCERTAIN_TAGS:
        if tag in line: return 'uncertain'
    return 'other'


def generate(matcher_output: dict, pair_meta: dict) -> dict:
    """matcher_output: TG record with 'rationale' key in canonical per-criterion format."""
    rationale = matcher_output.get('rationale') or matcher_output.get('cited_rationale') or ''
    inc_block, exc_block, inc_pass, exc_pass = [], [], [], []
    section = None
    for line in rationale.split('\n'):
        s = line.strip()
        if 'Inclusion criteria' in s and ':' in s: section = 'inc'; continue
        if 'Exclusion criteria' in s and ':' in s: section = 'exc'; continue
        if not s.startswith('- ['): continue
        cls = _classify(s, section or 'inc')
        if cls == 'blocker':
            (inc_block if section == 'inc' else exc_block).append('  ' + s)
        elif cls == 'passing':
            (inc_pass if section == 'inc' else exc_pass).append('  ' + s)
        # uncertain / other rows dropped from both lists

    parts = []
    if inc_block:
        parts.append('Inclusion criteria the patient FAILS:\n' + '\n'.join(inc_block))
    if exc_block:
        parts.append('Exclusion criteria the patient TRIGGERS:\n' + '\n'.join(exc_block))
    rationale_text = '\n\n'.join(parts) or '(no TG blockers extracted)'

    supports = []
    if inc_pass:
        supports.append('Inclusion criteria the patient already MEETS (preserve):\n' + '\n'.join(inc_pass))
    if exc_pass:
        supports.append('Exclusion criteria the patient already PASSES (preserve):\n' + '\n'.join(exc_pass))
    supports_text = '\n\n'.join(supports)

    return {
        'kind':      'rationale_text',
        'rationale': rationale_text,
        'supports':  supports_text,
        'source':    'trialgpt.blockers_only',
    }
