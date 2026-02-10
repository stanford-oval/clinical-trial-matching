#!/usr/bin/env python3
"""Apply arbiter-v2 overrides to the inclusion-side SMT programs and
re-solve with Z3 to compute patched verdicts. Then score against
the 5-judge gold and report the impact on AEGIS metrics.
"""
from __future__ import annotations
import json, pathlib, re
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[3]
V9   = ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v9_full'
V10  = ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v10_full'
CACHE = ROOT/'experiments/53_v2_full/arbiter_v2_cache'

import z3


def patch_program(program_lines, overrides):
    """Rewrite asserts on overridden atoms.
    overrides = list of {atom, new_value}."""
    if not overrides: return program_lines
    by_atom = {ov['atom']: ov['new_value'] for ov in overrides}
    out = []
    # Match assertions that constrain a specific atom: `(assert (= ATOM val))`,
    # `(assert (not ATOM))`, or `(assert ATOM)` (top-level boolean assertion).
    for line in program_lines:
        s = line.strip()
        m1 = re.match(r'^\(assert\s+\(=\s+([A-Za-z_][A-Za-z0-9_@]*)\s+(true|false|[\d.]+)\)\)$', s)
        m2 = re.match(r'^\(assert\s+\(not\s+([A-Za-z_][A-Za-z0-9_@]*)\)\)$', s)
        m3 = re.match(r'^\(assert\s+([A-Za-z_][A-Za-z0-9_@]*)\)$', s)
        atom = None
        if m1: atom = m1.group(1)
        elif m2: atom = m2.group(1)
        elif m3: atom = m3.group(1)
        # Also strip leading `patient_patient_` -> `patient_` mismatch (Z3 outputs both prefixes)
        atom_simple = re.sub(r'^patient_patient_', 'patient_', atom) if atom else None
        if atom and (atom in by_atom or (atom_simple and atom_simple in by_atom)):
            new_val = by_atom.get(atom, by_atom.get(atom_simple))
            if new_val is None:
                continue  # drop the assertion (atom becomes free)
            elif new_val is True:
                out.append(f'(assert (= {atom} true))')
            elif new_val is False:
                out.append(f'(assert (= {atom} false))')
            else:
                out.append(f'(assert (= {atom} {new_val}))')
        else:
            out.append(line)
    return out


def solve(program_text):
    s = z3.Solver()
    try:
        s.from_string(program_text)
    except Exception as e:
        return f'parse_error: {e}'
    res = s.check()
    return str(res)  # 'sat' / 'unsat' / 'unknown'


def main():
    gold = json.load((ROOT/'experiments/accuracy/data/gold_5sys.json').open())['gold']

    # Load all arbiter cache entries; index by (pair, full_tid, side)
    overrides_idx = defaultdict(list)
    cls_idx = {}
    for f in CACHE.glob('*.json'):
        try: o = json.loads(f.read_text())
        except: continue
        key = (o['pair'], o['full_tid'], o['side'])
        overrides_idx[key] = o.get('overrides') or []
        cls_idx[key] = o.get('classification_static','')

    # For each (pair, full_tid) collect inclusion patched-status
    patched_inc_status = {}  # (pair, full_tid) -> 'sat' | 'unsat'
    n_inc_patched = 0; n_inc_flipped = 0
    for (pair, full_tid, side), ovs in overrides_idx.items():
        if side != 'inclusion': continue
        if not ovs: continue
        # Find program lines (prefer v10, fall back to v9)
        pid = pair.split('__',1)[0]
        full_path = V10/pid/f'{full_tid}__full.json'
        if not full_path.exists():
            full_path = V9/pid/f'{full_tid}__full.json'
            if not full_path.exists(): continue
        full = json.load(full_path.open())
        prog = full.get('inclusion',{}).get('raw',{}).get('smt_program_lines') or []
        if not prog: continue
        patched = patch_program(prog, ovs)
        new_status = solve('\n'.join(patched))
        n_inc_patched += 1
        if new_status == 'sat':
            patched_inc_status[(pair, full_tid)] = 'sat'
            n_inc_flipped += 1
        else:
            patched_inc_status[(pair, full_tid)] = new_status

    print(f'Inclusion programs patched + re-solved: {n_inc_patched}')
    print(f'Flipped UNSAT → SAT: {n_inc_flipped}')

    # Build patched pair-level verdicts under asymmetric scheme
    # Inclusion: prefer arbiter-patched if available, else v10 status, else v9.
    # Exclusion: always v9.
    def load_status(d, pid, full_tid, side):
        f = d/pid/f'{full_tid}__{side}_stats.json'
        if not f.exists(): return None
        try: return json.loads(f.read_text()).get('status')
        except: return None

    aegis_arb = {}
    for pair, t in gold.items():
        pid, parent_nct = pair.split('__',1)
        # gather variant ids from v10 (full coverage)
        if not (V10/pid).exists(): continue
        for f in (V10/pid).glob(f'{parent_nct}*__overall.json'):
            full_tid = f.name.replace('__overall.json','')
            inc_st = patched_inc_status.get((pair, full_tid))
            if inc_st is None:
                inc_st = load_status(V10, pid, full_tid, 'inclusion')
            exc_st = load_status(V9, pid, full_tid, 'exclusion') or load_status(V10, pid, full_tid, 'exclusion')
            if inc_st == 'sat' and exc_st == 'sat':
                aegis_arb[pair] = True
            else:
                aegis_arb.setdefault(pair, False)

    # Score
    def score(preds):
        tp=fp=fn=tn=0
        for p,t in gold.items():
            if p not in preds: continue
            v = preds[p]
            if v and t: tp+=1
            elif v and not t: fp+=1
            elif not v and t: fn+=1
            else: tn+=1
        n=tp+fp+fn+tn
        P = tp/(tp+fp) if tp+fp else 0
        R = tp/(tp+fn) if tp+fn else 0
        F1 = 2*P*R/(P+R) if P+R else 0
        F2 = 5*P*R/(4*P+R) if 4*P+R else 0
        return n,P,R,F1,F2,tp,fp,fn,tn

    # Asymmetric (no arbiter v2) baseline for comparison
    def asym_no_arb():
        out = {}
        for pair, t in gold.items():
            pid, parent_nct = pair.split('__',1)
            if not (V10/pid).exists(): continue
            elig = False
            for f in (V10/pid).glob(f'{parent_nct}*__overall.json'):
                full_tid = f.name.replace('__overall.json','')
                inc_st = load_status(V10, pid, full_tid, 'inclusion') or load_status(V9, pid, full_tid, 'inclusion')
                exc_st = load_status(V9, pid, full_tid, 'exclusion')
                if inc_st == 'sat' and exc_st == 'sat':
                    elig = True; break
            out[pair] = elig
        return out

    asym = asym_no_arb()

    print()
    print(f'{"":<30}  n     P     R    F1    F2    tp  fp  fn  tn')
    for label, m in [('AEGIS asym (baseline)', asym), ('AEGIS asym + arbiter v2', aegis_arb)]:
        n,P,R,F1,F2,tp,fp,fn_,tn = score(m)
        print(f'{label:<30}  {n}  {P:.3f}  {R:.3f}  {F1:.3f}  {F2:.3f}   {tp:>3} {fp:>3} {fn_:>3} {tn:>3}')

    fns_recovered = [p for p in gold if gold[p] and asym.get(p) is False and aegis_arb.get(p) is True]
    print(f'\nFN→TP recoveries via arbiter v2: {len(fns_recovered)}')
    for p in fns_recovered: print(f'  - {p}')


if __name__ == '__main__':
    main()
