#!/usr/bin/env python3
"""Reproducibility script for the VERDICT two-tier headline.

Produces the verified headline table on disk:
  - silence-null bothsides             F1=0.835 P=0.763 R=0.923
  - + compiled patches (default tier)  F1=0.863 P=0.837 R=0.891
  - + opt-in LLM-only fallback (opt-in tier) F1=0.847 F2=0.920 R=0.976
  - + verdict-time pop-gate (variant)  F1=0.876 P=0.898 R=0.854
  - LLM-only (baseline)              F1=0.863 P=0.787 R=0.955
  - TrialGPT (baseline)                F1=0.375 P=0.82  R=0.24

Inputs (all already on disk):
  experiments/53h_v6_full/cmsrc_out/        v6 atom-mined trial-pair outputs
  experiments/53_v2_full/judges_<j>/        5-judge gold (clinician_v2 etc.)
  overnight/lm_population_gate.jsonl        LM gate verdicts (305 forwards)
  overnight/gate_to_constraint.jsonl        compiled per-pair patches (58 atoms)
  overnight/lm_only_V5_TWO_STEP.jsonl       LLM-only predictions
  overnight/baseline_metrics_trialgpt_matching.json  TrialGPT scores

Methodology: gold-majority of 5 judges (>=3/5 yes); silence-null applied to
both inclusion AND exclusion sides; compiled patches inject (assert atom=true)
with NULL->FALSE coercion on the patched atom only.

Run:
  python verdict/headline.py            (or: verdict headline)

Output: overnight/HEADLINE_VERIFIED.json with full table + provenance.
"""
from __future__ import annotations
import json, os, pathlib, re, sys
from collections import defaultdict
import z3

ROOT = pathlib.Path(os.environ.get(
    "VERDICT_ROOT", pathlib.Path(__file__).resolve().parents[1]))
DATA = ROOT / "data" / "headline"


# ============================================================================
# Doctrine: bothsides silence-null + per-pop-atom NULL->FALSE coercion
# ============================================================================

def is_silence(ev):
    el = (ev or '').lower()
    return (not el or 'silent' in el or 'defer' in el or 'no evidence' in el
            or 'not documented' in el or 'not mentioned' in el)


def quote_smt(name):
    return name if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name) else f'|{name}|'


def smt_number(v):
    """Format float for SMT-LIB without scientific notation."""
    f = float(v)
    if f == 0.0: return '0.0'
    s = f'{f:.20f}'.rstrip('0')
    if s.endswith('.'): s = s + '0'
    if s.startswith('-.'): s = '-0' + s[1:]
    if s.startswith('.'):  s = '0' + s
    return s


# THRESH atom binding: connect proxy boolean to the underlying numeric
# comparison. Without this, asserting a THRESH atom is logically meaningless.
_OP_MAP = {'ge':'>=', 'le':'<=', 'gt':'>', 'lt':'<', 'eq':'=', 'ne':'!='}
_THRESH_RE = re.compile(r'^__THRESH__::(.+)::(ge|le|gt|lt|eq|ne)::(-?\d+(?:\.\d+)?)$')

def thresh_binding_lines(atom_name, prog_lines=None):
    """Return SMT-LIB lines binding a THRESH atom to its numeric comparison.
    Skips redeclaration if the underlying variable is already declared in the
    program. Returns empty list if not a THRESH atom."""
    m = _THRESH_RE.match(atom_name)
    if not m: return []
    var, op_short, val = m.groups()
    qvar = quote_smt(var); qatom = quote_smt(atom_name)
    op = _OP_MAP[op_short]
    if op == '!=':
        binding = f'(not (= {qvar} {val}))'
    else:
        binding = f'({op} {qvar} {val})'
    out = [f'(declare-const {qatom} Bool)']
    # Only declare the underlying variable if not already declared
    if prog_lines is not None:
        already = any(re.search(rf'\(declare-const\s+{re.escape(var)}\s+', ln) or
                      re.search(rf'\(declare-fun\s+{re.escape(var)}\s+', ln)
                      for ln in prog_lines)
        if not already:
            out.append(f'(declare-const {qvar} Real)')
    out.append(f'(assert (= {qatom} {binding}))')
    return out


def solve(prog_lines, av, bool_pop):
    """Solve under bothsides silence-null + tagged-pop NULL->FALSE.
    THRESH atoms get bound to their underlying numeric comparison."""
    asserts = []; declared = set(); thresh_bound = set()
    extra_decls = []  # ordered list, may include declare-const + binding asserts
    declared_nonthresh = set()
    def bind_thresh(atom):
        if atom in thresh_bound: return
        if '__THRESH__' not in atom: return
        lines = thresh_binding_lines(atom, prog_lines=prog_lines)
        if lines:
            extra_decls.extend(lines)
            thresh_bound.add(atom)
    def ensure_declared(atom):
        if '__THRESH__' in atom: return
        if atom in declared_nonthresh: return
        already = any(re.search(rf'\(declare-(const|fun)\s+{re.escape(atom)}\s+', ln)
                      for ln in prog_lines)
        if not already:
            extra_decls.append(f'(declare-const {quote_smt(atom)} Bool)')
        declared_nonthresh.add(atom)
    for atom, rv in bool_pop.items():
        qn = quote_smt(atom)
        bind_thresh(atom); ensure_declared(atom)
        asserts.append(f'(assert (= {qn} {"true" if rv else "false"}))')
    pop_atoms = set(bool_pop)
    saw = set()
    for atom, m in av.items():
        if not isinstance(m, dict): continue
        saw.add(atom)
        v = m.get('value'); ev = m.get('evidence') or ''
        is_pop = atom in pop_atoms
        if v is None:
            if is_pop: v = False
            else: continue
        if v is False and is_silence(ev) and not is_pop:
            continue  # bothsides silence-null
        qn = quote_smt(atom)
        bind_thresh(atom)
        if isinstance(v, bool):
            asserts.append(f'(assert (= {qn} {"true" if v else "false"}))')
        elif isinstance(v, (int, float)):
            asserts.append(f'(assert (= {qn} {smt_number(v)}))')
        elif isinstance(v, str):
            sl = v.strip().lower()
            if sl == 'true': asserts.append(f'(assert (= {qn} true))')
            elif sl == 'false': asserts.append(f'(assert (= {qn} false))')
    for atom in pop_atoms:
        if atom not in saw:
            qn = quote_smt(atom)
            bind_thresh(atom); ensure_declared(atom)
            asserts.append(f'(assert (= {qn} false))')
    s = z3.Solver()
    try:
        s.from_string('\n'.join(prog_lines) + '\n' + '\n'.join(extra_decls)
                      + '\n' + '\n'.join(asserts))
        return s.check() == z3.sat
    except Exception:
        return None


# ============================================================================
# Data loading
# ============================================================================

def require_data():
    """Fail with guidance when the evaluation corpus is not in this build.

    The tool-only distribution ships the matcher without the paper's
    artifacts, so `verdict headline` cannot run there. Say where to get it
    rather than surfacing a FileNotFoundError from deep in the loader.
    """
    if (DATA / 'v6').is_dir():
        return
    raise SystemExit(
        "verdict headline reproduces the paper's table and needs the "
        "evaluation corpus, which is not part of this build.\n"
        f"Expected: {DATA / 'v6'}\n"
        "Point $VERDICT_ROOT at a checkout that has data/headline/, or use "
        "the full repository. See docs/DATA.md.")


def load_v6():
    require_data()
    v6 = defaultdict(dict)
    for pdir in (DATA/'v6').iterdir():
        if not pdir.is_dir() or pdir.name.startswith('_'): continue
        for fp in pdir.glob('*__full.json'):
            try: o = json.loads(fp.read_text())
            except: continue
            tid = fp.name.replace('__full.json','')
            m = re.match(r'^(NCT\d+)([a-z]?)$', tid)
            if not m: continue
            v6[f'{pdir.name}__{m.group(1)}'][m.group(2) or '_'] = (m.group(0), o)
    return v6


def load_gold():
    JUDGES = ['clinician_v2','clinician_paraphrase','engineering_canonical',
              'mechanical','rhetorical']
    def yes(s):
        return (s or '').strip().lower() in ('eligible','yes','forward','true','1')
    ppj = defaultdict(dict)
    for j in JUDGES:
        fp = DATA / 'judges' / f'{j}.jsonl'
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            try:
                o = json.loads(line)
                if o.get('pair'): ppj[o['pair']][j] = yes(o.get('judge_verdict'))
            except Exception:
                pass
    gold = {p: sum(d.values()) >= 3 for p, d in ppj.items() if len(d) == 5}
    return gold, ppj


def load_patches():
    """Load compiled per-pair population-fit patches."""
    patches = defaultdict(dict)
    for line in open(DATA/'gate_to_constraint.jsonl'):
        try: r = json.loads(line)
        except: continue
        if r.get('error') or not r.get('atom_name'): continue
        patches[r['pair']][r['atom_name']] = bool(r['required_value'])
    return patches


def load_gate_no():
    no = set()
    for line in open(DATA/'lm_population_gate.jsonl'):
        try: r = json.loads(line)
        except: continue
        if r.get('population_fit') == 'no': no.add(r['pair'])
    return no


def load_jsonl_preds(path):
    preds = {}
    for line in open(path):
        try: o = json.loads(line)
        except: continue
        preds[o['pair']] = (o.get('eligibility') == 'eligible')
    return preds


# ============================================================================
# Verdict computation
# ============================================================================

def compute_aegis_verdict(v6, patches=None):
    """Run VERDICT over all pairs with given patches (or empty)."""
    patches = patches or {}
    out = {}
    for pair, vs in v6.items():
        elig = False
        for key, (full_tid, vd) in vs.items():
            inc_raw = vd.get('inclusion',{}).get('raw',{}) or {}
            exc_raw = vd.get('exclusion',{}).get('raw',{}) or {}
            new_inc = solve(inc_raw.get('smt_program_lines') or [],
                            inc_raw.get('patient_var_values_rich') or {},
                            patches.get(pair, {}))
            new_exc = solve(exc_raw.get('smt_program_lines') or [],
                            exc_raw.get('patient_var_values_rich') or {},
                            {})
            old_inc = vd.get('inclusion',{}).get('sat_like')
            old_exc = vd.get('exclusion',{}).get('sat_like')
            inc_ok = new_inc if new_inc is not None else (old_inc is not False)
            exc_ok = new_exc if new_exc is not None else (old_exc is not False)
            if inc_ok and exc_ok:
                elig = True; break
        out[pair] = elig
    return out


# ============================================================================
# Metrics
# ============================================================================

def metrics(preds, gold):
    tp = fp = fn = tn = 0
    for p, gt in gold.items():
        if p not in preds: continue
        pr = preds[p]
        if pr and gt: tp += 1
        elif pr and not gt: fp += 1
        elif not pr and gt: fn += 1
        else: tn += 1
    n = tp + fp + fn + tn
    P = tp/(tp+fp) if (tp+fp) else 0
    R = tp/(tp+fn) if (tp+fn) else 0
    F1 = 2*P*R/(P+R) if (P+R) else 0
    F2 = 5*P*R/(4*P+R) if (4*P+R) else 0
    F05 = 1.25*P*R/(0.25*P+R) if (0.25*P+R) else 0
    return {'F1': F1, 'F2': F2, 'F0.5': F05, 'P': P, 'R': R,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'n': n}


# ============================================================================
# Main
# ============================================================================

def main():
    print('Loading inputs...')
    v6 = load_v6()
    gold, ppj = load_gold()
    patches = load_patches()
    gate_no = load_gate_no()
    print(f'  v6 pair-variants: {sum(len(v) for v in v6.values())} (over {len(v6)} pairs)')
    print(f'  gold-majority pairs: {len(gold)}')
    print(f'  compiled patches: {len(patches)} pairs')
    print(f'  gate-no rejections: {len(gate_no)}')

    # LLM-only (decision file already on disk)
    v5 = load_jsonl_preds(DATA/'lm_only_V5_TWO_STEP.jsonl')

    # Compute VERDICT verdicts
    print('\nComputing VERDICT verdicts...')
    verdict_silence = compute_aegis_verdict(v6, patches=None)
    verdict_default = compute_aegis_verdict(v6, patches=patches)
    print('  silence-null bothsides: done')
    print('  + compiled patches:     done')

    # Variants
    verdict_optin = {p: verdict_default.get(p, False) or v5.get(p, False)
                   for p in verdict_default if p in v5}
    verdict_gate = {p: (False if p in gate_no else verdict_silence[p])
                  for p in verdict_silence}

    rows = {
        'VERDICT+silence-null':      metrics(verdict_silence, gold),
        'VERDICT default (compiled)': metrics(verdict_default, gold),
        'VERDICT opt-in (∪LLM-only)':      metrics(verdict_optin, gold),
        'VERDICT+verdict-gate':      metrics(verdict_gate, gold),
        'LLM-only':              metrics(v5, gold),
    }

    # Format output
    print(f'\n{"":34s}  {"F1":>5s} {"F2":>5s} {"F0.5":>5s} {"P":>5s} {"R":>5s}  TP/FP/FN')
    for name, m in rows.items():
        print(f'  {name:32s}  {m["F1"]:.3f} {m["F2"]:.3f} {m["F0.5"]:.3f} '
              f'{m["P"]:.3f} {m["R"]:.3f}  {m["TP"]}/{m["FP"]}/{m["FN"]}')

    # Save provenance + numbers
    out = {
        'methodology': {
            'gold': 'gold-majority of 5 LM judges (>=3/5 yes)',
            'silence_null': 'bothsides (drops FALSE-on-silence on inclusion AND exclusion)',
            'compiled_patches_doctrine': 'NULL -> FALSE coercion on patched atoms only; (assert atom=true) injected',
        },
        'inputs': {
            'v6_dir': 'data/headline/v6  (reduced from the 53h_v6_full mining run; identical results)',
            'judge_dirs': 'data/headline/judges/<j>.jsonl for j in {clinician_v2, clinician_paraphrase, engineering_canonical, mechanical, rhetorical}',
            'gate_verdicts': 'data/headline/lm_population_gate.jsonl',
            'compiled_patches': 'data/headline/gate_to_constraint.jsonl',
            'llm_only_predictions': 'data/headline/lm_only_V5_TWO_STEP.jsonl',
        },
        'counts': {
            'pair_variants': sum(len(v) for v in v6.values()),
            'pairs': len(v6),
            'gold_majority_pairs': len(gold),
            'compiled_patches': 'data/headline/gate_to_constraint.jsonl',
            'gate_no_rejections': len(gate_no),
        },
        'rows': rows,
    }
    # Writing is opt-in. The upstream script wrote its result over the stored
    # HEADLINE_VERIFIED.json record; an assistant run destroyed that artifact
    # once already. Default to printing only.
    out_path = os.environ.get("HEADLINE_OUT")
    if not out_path:
        print("\n(not written; set $HEADLINE_OUT to save this table)")
        return
    out_path = pathlib.Path(out_path)
    out_path.write_text(json.dumps(out, indent=2))
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
