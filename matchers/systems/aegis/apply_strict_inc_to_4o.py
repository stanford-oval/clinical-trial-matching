#!/usr/bin/env python3
"""Apply asymmetric strict-inclusion doctrine on top of cmsrc 4o-mined data.

Loads the cmsrc per-pair full.json files, re-solves each side with:
  - inclusion side: silence-False atoms asserted as FALSE (strict)
  - exclusion side: silence-False atoms dropped (NULL)

Aggregates cohort variants (any-cohort eligible → eligible).

Output: jsonl with rich native_rationale rendered from patient_var_values_rich
plus the new strict-inc verdict.
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys
from collections import defaultdict
import z3

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'backup'))
try:
    from overnight.repro_headline import is_silence, quote_smt, smt_number, thresh_binding_lines
except ModuleNotFoundError as _e:  # pragma: no cover
    raise ModuleNotFoundError(
        "overnight is not part of this repository -- this batch driver "
        "depends on an internal module that was not released. The "
        "supported entry points are the `verdict` command and the "
        "`verdict` Python package; see README."
    ) from _e


def solve_strict_inc(prog_lines, av, side):
    asserts = []; extra_decls = []
    thresh_bound = set(); declared_nonthresh = set()

    def bind_thresh(atom):
        if atom in thresh_bound or '__THRESH__' not in atom: return
        lines = thresh_binding_lines(atom, prog_lines=prog_lines)
        if lines: extra_decls.extend(lines); thresh_bound.add(atom)

    def ensure_declared(atom):
        if '__THRESH__' in atom or atom in declared_nonthresh: return
        already = any(re.search(rf'\(declare-(const|fun)\s+{re.escape(atom)}\s+', ln) for ln in prog_lines)
        if not already:
            extra_decls.append(f'(declare-const {quote_smt(atom)} Bool)')
        declared_nonthresh.add(atom)

    for atom, m in (av or {}).items():
        if not isinstance(m, dict): continue
        v = m.get('value'); ev = m.get('evidence') or ''
        if v is None: continue
        if v is False and is_silence(ev) and side == 'exclusion':
            continue   # silence-null only on exclusion
        bind_thresh(atom); ensure_declared(atom)
        qn = quote_smt(atom)
        if isinstance(v, bool):
            asserts.append(f'(assert (= {qn} {"true" if v else "false"}))')
        elif isinstance(v, (int, float)):
            asserts.append(f'(assert (= {qn} {smt_number(v)}))')
        elif isinstance(v, str):
            sl = v.strip().lower()
            if sl == 'true': asserts.append(f'(assert (= {qn} true))')
            elif sl == 'false': asserts.append(f'(assert (= {qn} false))')

    s = z3.Solver()
    try:
        s.from_string('\n'.join(prog_lines) + '\n' + '\n'.join(extra_decls) + '\n' + '\n'.join(asserts))
        return s.check() == z3.sat
    except Exception:
        return None


def render_native(o, inc_strict, exc_strict):
    parts = []
    inc = (o.get('inclusion') or {}).get('raw') or {}
    exc = (o.get('exclusion') or {}).get('raw') or {}
    inc_pvr = inc.get('patient_var_values_rich') or {}
    exc_pvr = exc.get('patient_var_values_rich') or {}

    parts.append(f'Strict-inc inclusion result: {"SAT (no inclusion violation)" if inc_strict else "UNSAT (one or more inclusion criteria fail under strict-inc)"}')
    parts.append(f'Exclusion result: {"SAT (no exclusion fired)" if exc_strict else "UNSAT (an exclusion criterion was triggered)"}')

    parts.append('\nPer-atom inclusion mining:')
    for atom, info in list(inc_pvr.items())[:25]:
        if not isinstance(info, dict): continue
        ass = (info.get('assessment') or '')[:200]; ev = (info.get('evidence') or '')[:200]; val = info.get('value')
        parts.append(f'- {atom}: value={val} | assessment={ass} | evidence={ev}')

    parts.append('\nPer-atom exclusion mining:')
    for atom, info in list(exc_pvr.items())[:25]:
        if not isinstance(info, dict): continue
        ass = (info.get('assessment') or '')[:200]; ev = (info.get('evidence') or '')[:200]; val = info.get('value')
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

    variants = defaultdict(list)
    for pdir in sorted(mine.iterdir()):
        if not pdir.is_dir(): continue
        for fn in sorted(pdir.glob('NCT*__full.json')):
            tid = fn.name.replace('__full.json','')
            parent = re.sub(r'(?<=NCT\d{8})[a-z]+$','',tid)
            try: o = json.loads(fn.read_text())
            except: continue
            if 'error' in o: continue
            variants[(pdir.name, parent)].append({'tid': tid, 'o': o})

    print(f'logical pairs: {len(variants)}')

    n=0
    with out.open('w') as fout:
        for (pid, parent), vs in sorted(variants.items()):
            pair = f'{pid}__{parent}'
            elig_any = False
            chosen = None
            for v in vs:
                o = v['o']
                inc_raw = (o.get('inclusion') or {}).get('raw') or {}
                exc_raw = (o.get('exclusion') or {}).get('raw') or {}
                inc_lines = inc_raw.get('smt_program_lines') or []
                exc_lines = exc_raw.get('smt_program_lines') or []
                inc_av = inc_raw.get('patient_var_values_rich') or {}
                exc_av = exc_raw.get('patient_var_values_rich') or {}
                inc_strict = solve_strict_inc(inc_lines, inc_av, 'inclusion')
                exc_strict = solve_strict_inc(exc_lines, exc_av, 'exclusion')
                inc_ok = inc_strict if inc_strict is not None else True
                exc_ok = exc_strict if exc_strict is not None else True
                if inc_ok and exc_ok:
                    elig_any = True
                    chosen = (v, inc_ok, exc_ok)
                    break
            if chosen is None:
                # No eligible variant — pick first and record its strict-inc results
                v = vs[0]
                o = v['o']
                inc_raw = (o.get('inclusion') or {}).get('raw') or {}
                exc_raw = (o.get('exclusion') or {}).get('raw') or {}
                inc_strict = solve_strict_inc(inc_raw.get('smt_program_lines') or [],
                                              inc_raw.get('patient_var_values_rich') or {}, 'inclusion')
                exc_strict = solve_strict_inc(exc_raw.get('smt_program_lines') or [],
                                              exc_raw.get('patient_var_values_rich') or {}, 'exclusion')
                chosen = (v, inc_strict, exc_strict)

            v, inc_ok, exc_ok = chosen
            o = v['o']
            native = render_native(o, inc_ok, exc_ok)
            inc_status = 'SAT' if inc_ok else 'UNSAT'
            exc_status = 'SAT (no exclusion triggered)' if exc_ok else 'UNSAT (exclusion triggered)'
            rec = {
                'pair': pair,
                'eligibility': 'eligible' if elig_any else 'ineligible',
                'rationale': native,
                'deciding_variant': f'{v["tid"]} (asym strict-inc, any-cohort agg)',
                'arbiter_applied': False, 'arbiter_overrides': None,
                'inclusion_status': inc_status, 'exclusion_status': exc_status,
            }
            fout.write(json.dumps(rec)+'\n'); n+=1
    print(f'wrote {n} records → {out}')


if __name__ == '__main__':
    main()
