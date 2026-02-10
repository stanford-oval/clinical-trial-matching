#!/usr/bin/env python3
"""Convert cmsrc per-pair full.json files into a flat AEGIS rationales jsonl.

Aggregates cohort-variants per logical (patient, NCT-parent) pair: a patient
is eligible if ANY cohort variant of the trial is eligible (matches cmsrc
canonical aggregation).

Eligibility uses cmsrc's `eligible` field directly (inc_sat AND exc_sat).
"""
from __future__ import annotations
import argparse, json, pathlib, re
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[2]


def render_native_rationale(o):
    """Build a rich AEGIS native rationale from cmsrc per-atom mining output."""
    parts = []
    inc = o.get('inclusion', {}) or {}
    exc = o.get('exclusion', {}) or {}
    inc_pvr = (inc.get('raw') or {}).get('patient_var_values_rich') or {}
    exc_pvr = (exc.get('raw') or {}).get('patient_var_values_rich') or {}
    inc_summary = inc.get('summary') or {}
    exc_summary = exc.get('summary') or {}

    parts.append(f'AEGIS verdict: {"eligible" if o.get("eligible") else "ineligible"}.')
    parts.append(f'Inclusion side: {"SAT (no inclusion violation)" if inc.get("sat_like") else f"UNSAT — unsat_core: {inc_summary.get(chr(34)+chr(117)+chr(110)+chr(115)+chr(97)+chr(116)+chr(95)+chr(99)+chr(111)+chr(114)+chr(101)+chr(34),[])[:6]}"}.')
    parts.append(f'Exclusion side: {"SAT (no exclusion fired)" if exc.get("sat_like") else f"UNSAT — unsat_core: {exc_summary.get(chr(34)+chr(117)+chr(110)+chr(115)+chr(97)+chr(116)+chr(95)+chr(99)+chr(111)+chr(114)+chr(101)+chr(34),[])[:6]}"}.')

    parts.append('\nPer-atom inclusion mining:')
    for atom, info in list(inc_pvr.items())[:25]:
        if not isinstance(info, dict): continue
        ass = (info.get('assessment') or '')[:200]
        ev  = (info.get('evidence') or '')[:200]
        val = info.get('value')
        parts.append(f'- {atom}: value={val} | assessment={ass} | evidence={ev}')

    parts.append('\nPer-atom exclusion mining:')
    for atom, info in list(exc_pvr.items())[:25]:
        if not isinstance(info, dict): continue
        ass = (info.get('assessment') or '')[:200]
        ev  = (info.get('evidence') or '')[:200]
        val = info.get('value')
        parts.append(f'- {atom}: value={val} | assessment={ass} | evidence={ev}')

    return '\n'.join(parts)[:8000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mine-dir', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    mine = pathlib.Path(args.mine_dir)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # variants[(patient, parent_nct)] = list of {'tid','o','eligible'}
    variants = defaultdict(list)
    for pdir in sorted(mine.iterdir()):
        if not pdir.is_dir(): continue
        for fn in sorted(pdir.glob('NCT*__full.json')):
            tid = fn.name.replace('__full.json', '')
            parent = re.sub(r'(?<=NCT\d{8})[a-z]+$', '', tid)
            try:
                o = json.loads(fn.read_text())
            except: continue
            if 'error' in o: continue
            variants[(pdir.name, parent)].append({'tid': tid, 'o': o})

    print(f'logical pairs: {len(variants)}')

    n = 0
    with out.open('w') as fout:
        for (pid, parent), vs in sorted(variants.items()):
            pair = f'{pid}__{parent}'
            # Eligible if ANY variant is eligible (cmsrc convention)
            elig_any = any(v['o'].get('eligible') is True for v in vs)
            elig_strict_any = any(v['o'].get('eligible_strict') is True for v in vs)
            # Pick the first eligible variant if any, else the first
            chosen = next((v for v in vs if v['o'].get('eligible') is True), vs[0])
            o = chosen['o']
            inc = o.get('inclusion', {}) or {}
            exc = o.get('exclusion', {}) or {}
            inc_sat = inc.get('sat_like'); exc_sat = exc.get('sat_like')
            inc_status = 'SAT' if inc_sat else (
                f'UNSAT (cores: {(inc.get("summary") or {}).get("unsat_assertions",[])[:5]})')
            exc_status = 'SAT (no exclusion triggered)' if exc_sat else (
                f'UNSAT (exclusion triggered: {(exc.get("summary") or {}).get("unsat_assertions",[])[:5]})')
            native_rat = render_native_rationale(o)
            rec = {
                'pair': pair,
                'eligibility': 'eligible' if elig_any else 'ineligible',
                'rationale': native_rat,
                'deciding_variant': f'{chosen["tid"]} (any-cohort aggregation)',
                'arbiter_applied': False, 'arbiter_overrides': None,
                'inclusion_status': inc_status, 'exclusion_status': exc_status,
                'eligible_strict': elig_strict_any,
            }
            fout.write(json.dumps(rec) + '\n'); n += 1
    print(f'wrote {n} records → {out}')


if __name__ == '__main__':
    main()
