#!/usr/bin/env python3
"""Re-solve cmsrc per-pair output with PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE
auxiliaries treated as SOFT (skipped) under prescreen-doctrine.

Hypothesis: under prescreen, "all sub-conditions of a clinical-state criterion
must be observable in the chart" is too strict. Skip these auxiliaries; the
patient passes inclusion if all OTHER atoms are SAT (or NULL).
"""
from __future__ import annotations
import os
import argparse, json, pathlib, re, sys
from collections import defaultdict
sys.path.insert(0, str(pathlib.Path(os.environ.get('VERDICT_ROOT',
    pathlib.Path(__file__).resolve().parents[3])) / 'backup'))
try:
    from overnight.repro_headline import is_silence, quote_smt, smt_number, thresh_binding_lines
    import z3
except ModuleNotFoundError as _e:  # pragma: no cover
    raise ModuleNotFoundError(
        "overnight is not part of this repository -- this batch driver "
        "depends on an internal module that was not released. The "
        "supported entry points are the `verdict` command and the "
        "`verdict` Python package; see README."
    ) from _e

ROOT = pathlib.Path(os.environ.get('VERDICT_ROOT',
    pathlib.Path(__file__).resolve().parents[3]))

# Auxiliary :named patterns that we'll SKIP under prescreen-doctrine
SOFT_AUX_PATTERNS = [
    r'PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE',
    r'AUXILIARY\d*',  # other AUXILIARY clauses too
    r'OTHER_REQUIREMENTS',
]
SOFT_RE = re.compile('|'.join(SOFT_AUX_PATTERNS))


def filter_program(prog_lines):
    """Drop assertions whose :named tag matches SOFT_AUX_PATTERNS."""
    out = []
    for ln in prog_lines:
        m = re.search(r':named\s+([A-Za-z0-9_]+)', ln)
        if m and SOFT_RE.search(m.group(1)):
            continue   # skip this assertion
        out.append(ln)
    return out


def solve_side(prog_lines, av, side, soft_aux=True):
    if soft_aux:
        prog_lines = filter_program(prog_lines)
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
            continue
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mine-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--soft-aux', action='store_true', default=True,
                    help='skip PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE / AUXILIARY clauses (soft)')
    args = ap.parse_args()
    mine = pathlib.Path(args.mine_dir)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    variants = defaultdict(list)
    for pdir in sorted(mine.iterdir()):
        if not pdir.is_dir() or pdir.name.startswith('_'): continue
        for fn in sorted(pdir.glob('NCT*__full.json')):
            tid = fn.name.replace('__full.json','')
            parent = re.sub(r'(?<=NCT\d{8})[a-z]+$','',tid)
            try: o = json.loads(fn.read_text())
            except: continue
            if 'error' in o: continue
            variants[(pdir.name, parent)].append({'tid': tid, 'o': o})

    n=0
    with out.open('w') as fout:
        for (pid, parent), vs in sorted(variants.items()):
            pair = f'{pid}__{parent}'
            elig_any = False; chosen=None
            for v in vs:
                o = v['o']
                inc_raw = (o.get('inclusion') or {}).get('raw') or {}
                exc_raw = (o.get('exclusion') or {}).get('raw') or {}
                inc_ok = solve_side(inc_raw.get('smt_program_lines') or [],
                                    inc_raw.get('patient_var_values_rich') or {}, 'inclusion', args.soft_aux)
                exc_ok = solve_side(exc_raw.get('smt_program_lines') or [],
                                    exc_raw.get('patient_var_values_rich') or {}, 'exclusion', args.soft_aux)
                inc_ok = inc_ok if inc_ok is not None else True
                exc_ok = exc_ok if exc_ok is not None else True
                if inc_ok and exc_ok:
                    elig_any = True; chosen = v; break
            if chosen is None: chosen = vs[0]
            rec = {
                'pair': pair,
                'eligibility': 'eligible' if elig_any else 'ineligible',
                'rationale': f'soft-aux solve, deciding: {chosen["tid"]}',
                'deciding_variant': chosen['tid'],
            }
            fout.write(json.dumps(rec)+'\n'); n+=1
    print(f'wrote {n} → {out}')


if __name__ == '__main__':
    main()
