"""Shah Lab rationale generator: filter to blocker rows.

shah's Koopman output tags every criterion with `is_met: bool`. Uniform
convention: `is_met=true` = patient passes this criterion check; `is_met=false`
= patient fails this check = BLOCKER. This holds on both inclusion AND
exclusion sides.

So blockers = rows where `is_met=false`, regardless of side.

Drops [met] rows so the counterfactual modifier only sees actionable blockers
(prevents accidentally perturbing passing-criterion facts the patient already
satisfies).
"""
from __future__ import annotations


def generate(matcher_output: dict, pair_meta: dict) -> dict:
    """matcher_output: shah per-pair record with either:
       - 'assessments': [{criterion, rationale, is_met, confidence}, ...]   (raw output)
       - 'rationale': preformatted "Inclusion criteria (per-criterion verdict):\n  - [met] ..."
    """
    inc_blockers, exc_blockers, inc_passing, exc_passing = [], [], [], []

    assessments = matcher_output.get('assessments')
    if assessments:
        # Structured raw output path
        for a in assessments:
            crit_key = str(a.get('criterion', '')).lower()
            is_met = a.get('is_met')
            rat = (a.get('rationale','') or '').strip()
            tag = '[met]' if is_met else '[not met]'
            line = f'  - {tag} {rat}'
            on_inc = 'inclusion' in crit_key
            if is_met is False:
                (inc_blockers if on_inc else exc_blockers).append(line)
            else:
                (inc_passing if on_inc else exc_passing).append(line)
    else:
        # Pre-formatted rationale text path (fallback for legacy data)
        rat = matcher_output.get('rationale','') or ''
        section = None
        for line in rat.split('\n'):
            s = line.strip()
            if s.startswith('Inclusion criteria'): section = 'inc'; continue
            if s.startswith('Exclusion criteria'): section = 'exc'; continue
            if not s.startswith('- ['): continue
            on_inc = (section == 'inc')
            if '[not met]' in s[:14]:
                (inc_blockers if on_inc else exc_blockers).append('  ' + s)
            else:
                (inc_passing if on_inc else exc_passing).append('  ' + s)

    parts = []
    if inc_blockers:
        parts.append('Inclusion criteria the patient FAILS:\n' + '\n'.join(inc_blockers))
    if exc_blockers:
        parts.append('Exclusion criteria the patient TRIGGERS:\n' + '\n'.join(exc_blockers))
    rationale_text = '\n\n'.join(parts) or '(no shah blockers extracted)'

    supports = []
    if inc_passing:
        supports.append('Inclusion criteria the patient already MEETS (preserve):\n' + '\n'.join(inc_passing))
    if exc_passing:
        supports.append('Exclusion criteria the patient already PASSES (preserve):\n' + '\n'.join(exc_passing))
    supports_text = '\n\n'.join(supports)

    return {
        'kind':      'rationale_text',
        'rationale': rationale_text,
        'supports':  supports_text,
        'source':    'shahlab.blockers_only',
    }
