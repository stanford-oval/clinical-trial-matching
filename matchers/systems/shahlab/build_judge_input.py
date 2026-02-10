#!/usr/bin/env python3
"""Build stanford_for_judges.jsonl: per-criterion summary mirroring TG style.
Same render_rationale format used for trialgpt_for_judges.jsonl so the two
NL-baselines have comparable presentation to the gold judges.
"""
import os
import json, pathlib

ROOT = pathlib.Path(os.environ.get('VERDICT_ROOT',
    pathlib.Path(__file__).resolve().parents[3]))


def _clip_at_sentence(text: str, max_len: int = 500) -> str:
    """Trim to ≤ max_len chars but prefer to end at a sentence boundary.
    Avoids the previous bug where [:200] cut mid-clause (e.g., 'In fact, ')."""
    if not text: return ''
    text = text.strip()
    if len(text) <= max_len:
        return text
    # find the last sentence terminator within max_len
    window = text[:max_len]
    for i in range(len(window) - 1, max(0, max_len - 250) - 1, -1):
        if window[i] in '.!?' and (i == len(window) - 1 or window[i+1] in ' \n'):
            return window[: i + 1].strip()
    # no clean boundary — cut at last space and add ellipsis
    last_space = window.rfind(' ')
    if last_space > max_len - 80:
        return window[: last_space].rstrip() + '…'
    return window.rstrip() + '…'


def render_rationale(assessments, max_rows=8):
    """Render Stanford per-criterion assessments in TG-like format.
    Each criterion's rationale is clipped at a sentence boundary (≤500 chars)
    rather than hard-truncated, to avoid mid-clause cuts like 'In fact, '
    that confuse downstream CF rewriters and judges."""
    inc = [a for a in assessments if 'inclusion' in str(a.get('criterion','')).lower()]
    exc = [a for a in assessments if 'exclusion' in str(a.get('criterion','')).lower()]
    parts = []
    parts.append('Inclusion criteria (per-criterion Stanford verdict):')
    for a in inc[:max_rows]:
        met = a.get('is_met')
        label = 'met' if met else ('not met' if met is False else '?')
        parts.append(f"  - [{label}] {_clip_at_sentence(a.get('rationale') or '')}")
    parts.append('Exclusion criteria (per-criterion Stanford verdict):')
    for a in exc[:max_rows]:
        met = a.get('is_met')
        label = 'met' if met else ('not met' if met is False else '?')
        parts.append(f"  - [{label}] {_clip_at_sentence(a.get('rationale') or '')}")
    return '\n'.join(parts)


def main():
    src = ROOT/'overnight/stanford_som_shahlab.jsonl'
    out = ROOT/'overnight/stanford_for_judges.jsonl'
    rows = []
    with src.open() as f:
        for line in f:
            r = json.loads(line)
            asmts = r.get('assessments') or []
            rationale = render_rationale(asmts) if asmts else f"Stanford verdict {r.get('eligibility')} (global_decision={r.get('global_decision')}); per-criterion details unavailable."
            rows.append({
                'pair': r['pair'],
                'eligibility': r['eligibility'],
                'global_decision': r.get('global_decision'),
                'rationale': rationale[:2000],
                'n_inc_assessed': sum(1 for a in asmts if 'inclusion' in str(a.get('criterion','')).lower()),
                'n_exc_assessed': sum(1 for a in asmts if 'exclusion' in str(a.get('criterion','')).lower()),
            })
    with out.open('w') as f:
        for r in rows: f.write(json.dumps(r) + '\n')
    print(f'Wrote {out} with {len(rows)} pairs')
    print(f'Rationales: {sum(1 for r in rows if r["n_inc_assessed"] + r["n_exc_assessed"] > 0)} with per-criterion data')


if __name__ == '__main__':
    main()
