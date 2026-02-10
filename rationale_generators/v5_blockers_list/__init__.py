"""V5-with-blockers rationale generator: extract structured blockers[] array.

V5_TWO_STEP_BLOCKERS prompt returns:
  {
    "eligibility": "ineligible",
    "explanation": "<1-3 sentence summary>",
    "blockers": [{fact, side, criterion, reasoning}, ...],
    "supports": [{fact, side, criterion, reasoning}, ...]
  }

Modifier should only see `blockers` (actionable items) and a `supports`
hint of facts to preserve.
"""
from __future__ import annotations


def generate(matcher_output: dict, pair_meta: dict) -> dict:
    blockers = matcher_output.get('blockers') or []
    supports = matcher_output.get('supports') or []

    if blockers and isinstance(blockers[0], dict):
        lines = []
        for b in blockers:
            side = b.get('side','')
            fact = (b.get('fact') or '').strip()
            crit = (b.get('criterion') or '').strip()
            reason = (b.get('reasoning') or '').strip()
            lines.append(f'  - [{side}] criterion: {crit}\n    chart fact: {fact}\n    reasoning: {reason}')
        rationale_text = 'Patient blockers (must be addressed to flip the verdict):\n' + '\n'.join(lines)
    else:
        # Fallback for legacy records where blockers wasn't captured
        rationale_text = (matcher_output.get('explanation') or '').strip() or '(no V5_TWO_STEP_BLOCKERS blockers extracted)'

    if supports and isinstance(supports[0], dict):
        slines = []
        for s in supports:
            fact = (s.get('fact') or '').strip()
            crit = (s.get('criterion') or '').strip()
            slines.append(f'  - {fact}  (supports criterion: {crit})')
        supports_text = 'Patient supports already in place (preserve in CF):\n' + '\n'.join(slines)
    else:
        supports_text = ''

    return {
        'kind':      'rationale_text',
        'rationale': rationale_text,
        'supports':  supports_text,
        'source':    'single_shot_llm.v5_blockers_list',
    }
