#!/usr/bin/env python3
"""Phase-2 evaluator: load arbiter cache + re-solve with overrides applied.

Runs Z3 in-process but sequentially (worker count = 1) since multi-threaded Z3
can abort. For each (pair, variant), if the arbiter cache has overrides for
either side, apply them and re-solve. Otherwise use cmsrc's stored sat_like.
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys
from collections import defaultdict
import z3

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/accuracy/scripts'))
from repro_headline import quote_smt, smt_number, thresh_binding_lines, metrics


def solve_with_overrides(prog_lines, av, overrides):
    """Apply override values to LLM-assigned atoms; solve. Returns True/False/None."""
    if not prog_lines: return None
    drop_map = {o['atom']: o['new_value'] for o in overrides or []}
    asserts = []
    declared_nonthresh = set(); thresh_bound = set(); extra_decls = []
    def bind_thresh(atom):
        if atom in thresh_bound or '__THRESH__' not in atom: return
        lines = thresh_binding_lines(atom, prog_lines=prog_lines)
        if lines: extra_decls.extend(lines); thresh_bound.add(atom)
    def ensure_declared(atom):
        if '__THRESH__' in atom or atom in declared_nonthresh: return
        already = any(re.search(rf'\(declare-(const|fun)\s+{re.escape(atom)}\s+', ln) for ln in prog_lines)
        if not already: extra_decls.append(f'(declare-const {quote_smt(atom)} Bool)')
        declared_nonthresh.add(atom)
    for atom, m in (av or {}).items():
        if not isinstance(m, dict): continue
        if atom in drop_map:
            v = drop_map[atom]
            if v is None: continue
            qn = quote_smt(atom); bind_thresh(atom); ensure_declared(atom)
            if isinstance(v, bool):
                asserts.append(f'(assert (= {qn} {"true" if v else "false"}))')
            continue
        v = m.get('value')
        if v is None: continue
        qn = quote_smt(atom); bind_thresh(atom); ensure_declared(atom)
        if isinstance(v, bool):
            asserts.append(f'(assert (= {qn} {"true" if v else "false"}))')
        elif isinstance(v, (int,float)):
            asserts.append(f'(assert (= {qn} {smt_number(v)}))')
        elif isinstance(v, str):
            sl = v.strip().lower()
            if sl=='true': asserts.append(f'(assert (= {qn} true))')
            elif sl=='false': asserts.append(f'(assert (= {qn} false))')
    try:
        s = z3.Solver()
        s.from_string('\n'.join(prog_lines)+'\n'+'\n'.join(extra_decls)+'\n'+'\n'.join(asserts))
        return s.check() == z3.sat
    except Exception:
        return None


def load_cache(cache_dir):
    """Load arbiter overrides keyed by (pair, full_tid, side)."""
    cache = {}
    for fp in pathlib.Path(cache_dir).glob('*__inclusion.json'):
        # Filename: <pair>__<full_tid>__inclusion.json
        # pair has __ in it (sigir-XXX__NCTXXXXX), so split from the right.
        stem = fp.stem  # drops .json
        parts = stem.rsplit('__', 2)
        if len(parts) != 3 or parts[2] != 'inclusion': continue
        pair = parts[0]; full_tid = parts[1]
        try:
            cache[(pair, full_tid, 'inclusion')] = json.loads(fp.read_text())
        except: pass
    for fp in pathlib.Path(cache_dir).glob('*__exclusion.json'):
        stem = fp.stem
        parts = stem.rsplit('__', 2)
        if len(parts) != 3 or parts[2] != 'exclusion': continue
        pair = parts[0]; full_tid = parts[1]
        try:
            cache[(pair, full_tid, 'exclusion')] = json.loads(fp.read_text())
        except: pass
    return cache


def load_v9(mine_dir):
    out = defaultdict(dict)
    for pdir in pathlib.Path(mine_dir).iterdir():
        if not pdir.is_dir() or pdir.name.startswith('_'): continue
        for fp in pdir.glob('*__full.json'):
            try: o = json.loads(fp.read_text())
            except: continue
            tid = fp.name.replace('__full.json','')
            m = re.match(r'^(NCT\d+)([a-z]?)$', tid)
            if not m: continue
            out[f'{pdir.name}__{m.group(1)}'][m.group(2) or '_'] = (m.group(0), o)
    return out


def load_gold():
    JUDGES = ['clinician_v2','clinician_paraphrase','engineering_canonical','mechanical','rhetorical']
    def yes(s): return (s or '').strip().lower() in ('eligible','yes','forward','true','1')
    ppj = defaultdict(dict)
    for j in JUDGES:
        for fp in (ROOT/f'backup/experiments/53_v2_full/judges_{j}').glob('*.json'):
            try:
                o = json.loads(fp.read_text())
                if o.get('pair'): ppj[o['pair']][j] = yes(o.get('judge_verdict'))
            except: pass
    return {p: sum(d.values()) >= 3 for p, d in ppj.items() if len(d) == 5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mine-dir', default=str(ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v9_full'))
    ap.add_argument('--cache-dir', default=str(ROOT/'experiments/53_v2_full/arbiter_cache'))
    args = ap.parse_args()

    print(f'Loading mine + arbiter cache...')
    mine = load_v9(args.mine_dir)
    gold = load_gold()
    cache = load_cache(args.cache_dir)
    print(f'  pairs: {len(mine)}, gold: {len(gold)}')
    print(f'  arbiter cache entries: {len(cache)}')

    raw_verdicts = {}; arbiter_verdicts = {}
    n_with_overrides = 0; n_flips = 0
    for pair, vs in mine.items():
        if pair not in gold: continue
        raw_elig = False; arb_elig = False
        for key, (full_tid, vd) in vs.items():
            inc_raw = vd.get('inclusion',{}).get('raw',{}) or {}
            exc_raw = vd.get('exclusion',{}).get('raw',{}) or {}
            inc_prog = inc_raw.get('smt_program_lines') or []
            exc_prog = exc_raw.get('smt_program_lines') or []
            inc_av = inc_raw.get('patient_var_values_rich') or {}
            exc_av = exc_raw.get('patient_var_values_rich') or {}
            oi = vd.get('inclusion',{}).get('sat_like'); oe = vd.get('exclusion',{}).get('sat_like')
            # Raw verdict: cmsrc sat_like
            ri = oi is not False; re_ = oe is not False
            if ri and re_: raw_elig = True
            # Arbiter-applied verdict
            inc_cache = cache.get((pair, full_tid, 'inclusion')) or {}
            exc_cache = cache.get((pair, full_tid, 'exclusion')) or {}
            inc_overrides = inc_cache.get('overrides') or []
            exc_overrides = exc_cache.get('overrides') or []
            if inc_overrides or exc_overrides: n_with_overrides += 1
            if oi is False and inc_overrides:
                ai = solve_with_overrides(inc_prog, inc_av, inc_overrides)
                ai = ai if ai is not None else False
            else:
                ai = ri
            if oe is False and exc_overrides:
                ae = solve_with_overrides(exc_prog, exc_av, exc_overrides)
                ae = ae if ae is not None else False
            else:
                ae = re_
            if ai and ae: arb_elig = True
        raw_verdicts[pair] = raw_elig
        arbiter_verdicts[pair] = arb_elig
        if raw_elig != arb_elig: n_flips += 1

    g_raw = {p: gold[p] for p in raw_verdicts}
    g_arb = {p: gold[p] for p in arbiter_verdicts}
    mr = metrics(raw_verdicts, g_raw); ma = metrics(arbiter_verdicts, g_arb)
    print(f'\nVariants with arbiter overrides applied: {n_with_overrides}')
    print(f'Pair-level verdict flips (raw → arbiter): {n_flips}\n')
    print(f'{"variant":24s}  {"F1":>5s} {"P":>5s} {"R":>5s}   TP/FP/FN  n')
    print(f'  {"v9 raw":22s}  {mr["F1"]:.3f} {mr["P"]:.3f} {mr["R"]:.3f}   {mr["TP"]}/{mr["FP"]}/{mr["FN"]}  {mr["n"]}')
    print(f'  {"v9 + arbiter":22s}  {ma["F1"]:.3f} {ma["P"]:.3f} {ma["R"]:.3f}   {ma["TP"]}/{ma["FP"]}/{ma["FN"]}  {ma["n"]}')

    # Diff analysis
    flips_FT = [p for p in raw_verdicts if not raw_verdicts[p] and arbiter_verdicts.get(p)]
    flips_TF = [p for p in raw_verdicts if raw_verdicts[p] and not arbiter_verdicts.get(p)]
    print(f'\nFlips F→T: {len(flips_FT)}  (correct: {sum(1 for p in flips_FT if gold[p])})')
    print(f'Flips T→F: {len(flips_TF)}  (correct: {sum(1 for p in flips_TF if not gold[p])})')


if __name__ == '__main__':
    main()
