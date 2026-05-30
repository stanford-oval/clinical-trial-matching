#!/usr/bin/env python3
"""Reproduce the per-system verdict balance and rationale length table
(paper Appendix G, "Per-system verdict balance and rationale length").

Canonical sources are the *_freeform.jsonl artifacts -- these are what the
paper was computed from, verified 2026-09-03 by exact match on both the
eligible-rate and the median rationale length:

    VERDICT   aegis/aegis_freeform.jsonl              55.4%  / 1021 chars
    ourLLM    single_shot_llm/v5_freeform.jsonl       35.0%  / 1195 chars
    CoT LLM   single_shot_llm/v5_verbose_v2_freeform  36.1%  /  592 chars

(An earlier version of this script read verdicts.jsonl and a nonexistent
lm_only_V5_TWO_STEP.jsonl, and reproduced none of the paper's numbers.)

ZSPM is close but not exact (54.1% vs the paper's 54.0%): shahlab/rationales
covers 538 of the 552 pairs, so the paper's figure likely comes from a
fuller run, and it is flagged in the output rather than silently reported.

The paper's TrialGPT row is not reproduced here. No artifact in this
repository matched it (13.2% eligible against the paper's 44.4%), and the
TrialGPT-derived code and artifacts have since been removed altogether --
see docs/MATCHERS.md, "Why the TrialGPT baseline is not shipped". Cite
TrialGPT from its own repository, not from this table.

Usage:
    python scripts/tables/table10_verdict_balance.py
"""
from __future__ import annotations
import json, pathlib, statistics

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLD = ROOT / 'data/gold/gold_5sys_freeform_balanced.json'

# label -> (file, paper eligible-%, paper median chars, status)
SYSTEMS = [
    ('VERDICT',  'matchers/systems/aegis/aegis_freeform.jsonl',                   55.4, 1021, 'exact'),
    ('ourLLM',   'matchers/systems/single_shot_llm/v5_freeform.jsonl',            35.0, 1195, 'exact'),
    ('CoT LLM',  'matchers/systems/single_shot_llm/v5_verbose_v2_freeform.jsonl', 36.1,  592, 'exact'),
    ('ZSPM',     'matchers/systems/shahlab/rationales.jsonl',                     54.0, 1342, 'near'),
]

RATIONALE_KEYS = ('verbalized_rationale', 'rationale', 'text',
                  'explanation', 'key_points', 'reasoning')


def load_jsonl(fp: pathlib.Path) -> list:
    out = []
    if not fp.exists():
        return out
    for line in fp.open():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def rationale_len(rec) -> int:
    for k in RATIONALE_KEYS:
        v = rec.get(k)
        if isinstance(v, str) and v:
            return len(v)
        if isinstance(v, list) and v:
            return len(' '.join(str(x) for x in v))
    return 0


def main():
    gold = set(json.loads(GOLD.read_text())['gold']) if GOLD.exists() else None
    if gold:
        print(f'restricted to the {len(gold)}-pair comparison set\n')

    hdr = (f'{"system":<10s}{"n":>5s}{"elig%":>8s}{"paper":>8s}'
           f'{"med":>7s}{"paper":>7s}  status')
    print(hdr)
    print('-' * len(hdr))
    notes = []
    for label, rel, p_elig, p_med, status in SYSTEMS:
        recs = load_jsonl(ROOT / rel)
        if not recs:
            print(f'{label:<10s}  MISSING: {rel}')
            continue
        scored = [r for r in recs if gold is None or r.get('pair') in gold]
        n_e = sum(1 for r in scored
                  if str(r.get('eligibility', '')).lower() == 'eligible')
        elig = 100 * n_e / max(1, len(scored))
        lens = [x for x in (rationale_len(r) for r in recs) if x > 0]
        med = statistics.median(lens) if lens else 0
        flag = {'exact': 'ok', 'near': 'near', 'MISMATCH': 'DOES NOT MATCH'}[status]
        print(f'{label:<10s}{len(scored):>5d}{elig:>8.1f}{p_elig:>8.1f}'
              f'{med:>7.0f}{p_med:>7.0f}  {flag}')
        if status == 'near':
            notes.append(f'{label}: covers {len(scored)} of 552 pairs; '
                         f'paper likely used a fuller run.')
        if status == 'MISMATCH':
            notes.append(f'{label}: no artifact in this repo reproduces the '
                         f'paper column ({elig:.1f}% vs {p_elig}%). Do not cite '
                         f'this row.')
    if notes:
        print('\nnotes:')
        for nline in notes:
            print(f'  - {nline}')


if __name__ == '__main__':
    main()
