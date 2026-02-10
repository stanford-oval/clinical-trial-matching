#!/usr/bin/env python3
"""Compute AEGIS verdicts with NO silence-null on inclusion side.

Inclusion side: silence-null doctrine OFF — silent-False atoms asserted as FALSE (strict).
Exclusion side: silence-null doctrine ON — silent-False atoms dropped (silence not informative).
No compiled patches.

This is the variant the user requested ('we shouldn't have silence null on inclusion').
"""
from __future__ import annotations
import json, pathlib, re, sys
import z3

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from overnight.repro_headline import (
        is_silence, quote_smt, smt_number, thresh_binding_lines,
        load_v6, load_gold, metrics,
    )
except ModuleNotFoundError as _e:  # pragma: no cover
    raise ModuleNotFoundError(
        "overnight is not part of this repository -- this batch driver "
        "depends on an internal module that was not released. The "
        "supported entry points are the `verdict` command and the "
        "`verdict` Python package; see README."
    ) from _e


def solve_strict_inc(prog_lines, av, side, bool_pop=None):
    """Solve a single side. Silence-null applied ONLY on exclusion side."""
    bool_pop = bool_pop or {}
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

    for atom, rv in bool_pop.items():
        bind_thresh(atom); ensure_declared(atom)
        qn = quote_smt(atom)
        asserts.append(f'(assert (= {qn} {"true" if rv else "false"}))')
    pop_atoms = set(bool_pop)

    saw = set()
    for atom, m in av.items():
        if not isinstance(m, dict): continue
        saw.add(atom)
        v = m.get('value'); ev = m.get('evidence') or ''
        if atom in pop_atoms: continue
        if v is None:
            # Null value: drop assertion (no info)
            continue
        if v is False and is_silence(ev):
            # Silence-null: only drop on exclusion side
            if side == 'exclusion':
                continue
            # inclusion side: assert false (strict)
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
        s.from_string('\n'.join(prog_lines) + '\n' + '\n'.join(extra_decls)
                      + '\n' + '\n'.join(asserts))
        return s.check() == z3.sat
    except Exception:
        return None


def main():
    print('=== AEGIS strict-inclusion (no silence-null on inclusion) ===\n')

    v6 = load_v6()
    gold, _ = load_gold()
    print(f'v6: {len(v6)}  gold: {len(gold)}')

    out_path = ROOT / 'overnight/aegis_strict_inc.jsonl'
    verdicts = {}
    for pair, vs in v6.items():
        elig = False
        for key in sorted(vs):
            full_tid, vd = vs[key]
            inc_raw = vd.get('inclusion', {}).get('raw', {}) or {}
            exc_raw = vd.get('exclusion', {}).get('raw', {}) or {}
            inc = solve_strict_inc(inc_raw.get('smt_program_lines') or [],
                                   inc_raw.get('patient_var_values_rich') or {},
                                   side='inclusion')
            exc = solve_strict_inc(exc_raw.get('smt_program_lines') or [],
                                   exc_raw.get('patient_var_values_rich') or {},
                                   side='exclusion')
            old_inc = vd.get('inclusion', {}).get('sat_like')
            old_exc = vd.get('exclusion', {}).get('sat_like')
            inc_ok = inc if inc is not None else (old_inc is not False)
            exc_ok = exc if exc is not None else (old_exc is not False)
            if inc_ok and exc_ok:
                elig = True; break
        verdicts[pair] = elig

    with out_path.open('w') as f:
        for pair, elig in sorted(verdicts.items()):
            f.write(json.dumps({'pair': pair,
                               'eligibility': 'eligible' if elig else 'ineligible'}) + '\n')
    print(f'\nWrote {out_path}')
    print(f'Verdicts: {sum(1 for v in verdicts.values() if v)} eligible, '
          f'{sum(1 for v in verdicts.values() if not v)} ineligible')

    # Score
    m = metrics(verdicts, gold)
    print(f'\nvs old 5-judge gold:    F1={m["F1"]:.3f} P={m["P"]:.3f} R={m["R"]:.3f}')

    refined = ROOT/'overnight/gold_refined_5judges_verbalized.json'
    if refined.exists():
        new_gold = {p: bool(v) for p, v in json.loads(refined.read_text())['gold'].items()}
        m2 = metrics(verdicts, new_gold)
        print(f'vs refined gold:        F1={m2["F1"]:.3f} P={m2["P"]:.3f} R={m2["R"]:.3f}')

    # Compare to other variants
    print('\n=== Comparison ===')
    for fname, label in [
        ('aegis_silence_null_BOTHSIDES.jsonl', 'silence-null bothsides'),
        ('aegis_compiled_BOTHSIDES.jsonl', 'silence-null + patches'),
        ('aegis_strict_inc.jsonl', 'STRICT-INCLUSION (this variant)'),
    ]:
        path = ROOT/'overnight'/fname
        if not path.exists(): continue
        preds = {}
        for line in path.open():
            r = json.loads(line)
            preds[r['pair']] = (r['eligibility'] == 'eligible')
        m_old = metrics(preds, gold)
        if refined.exists():
            m_new = metrics(preds, new_gold)
            print(f'  {label:35s}  old: F1={m_old["F1"]:.3f}/{m_old["R"]:.3f}R  '
                  f'refined: F1={m_new["F1"]:.3f}/{m_new["R"]:.3f}R')
        else:
            print(f'  {label:35s}  F1={m_old["F1"]:.3f} R={m_old["R"]:.3f}')


if __name__ == '__main__':
    main()
