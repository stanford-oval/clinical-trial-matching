#!/usr/bin/env python3
"""AEGIS Arbiter — conflict resolution stage.

When the SMT solver returns UNSAT for a (pair, variant), we run a static
detector to classify the failure. If the unsat core contains:
  - explicit AUXILIARY / AUTO_AUXILIARY tags, OR
  - program-defined atoms (LHS of `(assert (= X (compound)))`), OR
  - same-atom-family inconsistencies (X_now vs X_inthehistory, qualifier vs stem)

…then the failure is likely an LLM atom-extraction inconsistency rather than a
genuine ineligibility. The arbiter sends (chart + offending atoms + relevant
constraint text) to an LLM to decide which atom value to flip / null. We re-
solve with the resolution applied.

Cache: experiments/53_v2_full/arbiter_cache/<pair>__<variant>__<side>.json
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, sys, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import z3

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/accuracy/scripts'))
from repro_headline import quote_smt, smt_number, thresh_binding_lines, is_silence, metrics

ENDPOINT = os.environ.get('OPENAI_ENDPOINT', '')
KEY = os.environ.get('OPENAI_API_KEY', '')

DEF_RE = re.compile(r'^\s*\(assert\s*(?:\(!\s*)?\(=\s*\|?([A-Za-z_][A-Za-z0-9_@]*)\|?\s*\(', re.MULTILINE)
AUX_RE = re.compile(r'AUXILIARY|AUTO_AUXILIARY', re.IGNORECASE)


ARBITER_PROMPT = """You are an expert clinician auditing an LLM's atomic extraction of patient facts.

The SMT solver reports the following atoms form a CONTRADICTION (UNSAT core).
The contradiction is internal to the LLM's extractions — the program has
auxiliary/definitional constraints that link these atoms (e.g., state continuity
between `X_now` and `X_inthehistory`; qualifier-implies-stem; aggregate counts
defined by constituents).

Your job: review each LLM-assigned atom in the core against the chart, and
output the minimal set of value changes that resolves the contradiction while
faithfully reflecting the chart.

PATIENT CHART:
{chart}

CONTRADICTING ATOMS (LLM-assigned values + evidence):
{atoms_listing}

RELEVANT PROGRAM CONSTRAINTS (auxiliary / definitional assertions in the unsat core):
{program_assertions}

Output STRICT JSON, no commentary:
{{
  "overrides": [
    {{"atom": "<atom_name>", "new_value": true | false | null,
      "rationale": "<one sentence: why the chart supports this value>"}},
    ...
  ]
}}

Rules:
- Prefer NULL when the chart truly doesn't speak to the atom (defer to visit).
- Prefer flipping a value (true↔false) only when the chart provides clear
  evidence. State continuity: if a finding is `X_now=true` and the chart
  describes a recent/acute presentation, `X_inthehistory` is likely also true
  (recent onset is part of history). Qualifier-stem: a positive stem with a
  silent qualifier should leave the qualifier NULL, not FALSE.
- Output the MINIMAL set of overrides needed. Do not list atoms whose LLM value
  is correct.
- If you cannot determine the right resolution from the chart, output
  `"overrides": []` and the solver will return UNSAT (patient ineligible)."""


# ---------- core SMT helpers ----------

def build_with_drop(prog_lines, av, drop=None):
    """Build (declarations + binding extras + named asserts list) for solving.
    Returns (parts, named_atoms) where named_atoms maps name → atom_name."""
    drop = drop or {}  # {atom_name: new_value or None}
    declared_nonthresh = set(); thresh_bound = set(); extra_decls = []
    asserts = []  # (atom_name, smt_assertion_str)
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
        if atom in drop:
            new_v = drop[atom]
            if new_v is None: continue  # null override = drop assertion
            qn = quote_smt(atom); bind_thresh(atom); ensure_declared(atom)
            if isinstance(new_v, bool):
                asserts.append((atom, f'(= {qn} {"true" if new_v else "false"})'))
            continue
        v = m.get('value')
        if v is None: continue
        qn = quote_smt(atom); bind_thresh(atom); ensure_declared(atom)
        if isinstance(v, bool):
            asserts.append((atom, f'(= {qn} {"true" if v else "false"})'))
        elif isinstance(v, (int,float)):
            asserts.append((atom, f'(= {qn} {smt_number(v)})'))
        elif isinstance(v, str):
            sl = v.strip().lower()
            if sl=='true': asserts.append((atom, f'(= {qn} true)'))
            elif sl=='false': asserts.append((atom, f'(= {qn} false)'))
    return prog_lines, extra_decls, asserts


def _solve_sat_only(prog_lines, av, drop=None):
    """Cheap SAT check without named asserts / core extraction."""
    if not prog_lines: return None
    prog, extras, asserts = build_with_drop(prog_lines, av, drop)
    parts = list(prog) + extras
    for atom, ass in asserts:
        parts.append(f'(assert {ass})')
    try:
        s = z3.Solver()
        s.from_string('\n'.join(parts))
        r = s.check()
    except Exception:
        return None
    if r == z3.sat: return True
    if r == z3.unsat: return False
    return None


def solve_named(prog_lines, av, drop=None):
    """Two-step: cheap SAT check first, then named-asserts re-solve only on UNSAT."""
    if not prog_lines: return None, None
    quick = _solve_sat_only(prog_lines, av, drop)
    if quick is None: return None, None
    if quick is True: return True, None
    # UNSAT: re-solve with named asserts to get core
    prog, extras, asserts = build_with_drop(prog_lines, av, drop)
    parts = ['(set-option :produce-unsat-cores true)'] + list(prog) + extras
    name_to_atom = {}
    for i, (atom, ass) in enumerate(asserts):
        name = f'pa_{i}'
        name_to_atom[name] = atom
        parts.append(f'(assert (! {ass} :named {name}))')
    try:
        s = z3.Solver()
        s.from_string('\n'.join(parts))
        r = s.check()
        if r != z3.unsat: return False, ([], [])  # fall back: no core info
        core = [str(c) for c in s.unsat_core()]
    except Exception:
        return False, ([], [])
    atoms_in_core = [name_to_atom[c] for c in core if c in name_to_atom]
    program_asserts_in_core = [c for c in core if c not in name_to_atom]
    return False, (atoms_in_core, program_asserts_in_core)


def is_arbiter_case(prog_lines, atoms_in_core, program_asserts_in_core):
    """Decide whether this UNSAT triggers arbiter resolution."""
    # 1. Explicit AUX tags in program-internal portion of core
    for assertion_name in program_asserts_in_core:
        if AUX_RE.search(assertion_name): return True, 'aux_tag'
    # 2. Program-defined atoms in core
    defined = set(DEF_RE.findall('\n'.join(prog_lines)))
    if any(a in defined for a in atoms_in_core): return True, 'defined_atom'
    # 3. Same-family inconsistency: X_now ↔ X_inthehistory or stem ↔ qualifier
    families = defaultdict(list)
    for a in atoms_in_core:
        # strip suffix to find family
        family = re.sub(r'(@@\w+)?(_now|_inthehistory|_inthepast\w*)?$', '', a)
        families[family].append(a)
    for fam, atoms in families.items():
        if len(atoms) >= 2: return True, 'family_inconsistency'
    return False, None


# ---------- arbiter LLM call ----------

def get_relevant_program_text(prog_lines, program_asserts_in_core, max_lines=20):
    """Extract the program lines mentioning the named auxiliary assertions."""
    out = []
    for ln in prog_lines:
        if any(name in ln for name in program_asserts_in_core):
            out.append(ln.strip()[:300])
        if len(out) >= max_lines: break
    return '\n'.join(out) if out else '(no specific auxiliary text extracted)'


def call_arbiter(chart, atoms_with_values, program_text):
    """Returns list of {atom, new_value, rationale}."""
    if not atoms_with_values:
        return []
    listing = '\n'.join(
        f'- `{a["atom"]}` = {a["value"]}\n  evidence: {(a["evidence"] or "")[:200]}\n  assessment: {(a.get("assessment") or "")[:200]}'
        for a in atoms_with_values
    )
    prompt = ARBITER_PROMPT.format(
        chart=(chart or '')[:3000],
        atoms_listing=listing,
        program_assertions=program_text,
    )
    body = json.dumps({
        'messages': [{'role':'user','content':prompt}],
        'max_tokens': 800, 'temperature': 0,
        'response_format': {'type': 'json_object'},
    }).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/chat/completions?api-version=2024-08-01-preview",
        data=body, headers={'api-key': KEY, 'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = json.loads(r.read())
        out = json.loads(resp['choices'][0]['message']['content'])
        return out.get('overrides') or []
    except Exception as e:
        print(f'  arbiter ERR: {e}', file=sys.stderr)
        return []


# ---------- per-pair processing ----------

def heuristic_candidates(prog_lines, av):
    """Without Z3 unsat-core, heuristically pick LLM atoms that may be involved in
    a definitional / family conflict. Returns (candidate_atoms, program_text)."""
    prog_text = '\n'.join(prog_lines)
    has_aux = bool(AUX_RE.search(prog_text))
    defined = set(DEF_RE.findall(prog_text))
    candidates = []
    # 1. Defined atoms with LLM-assigned values
    for atom, m in (av or {}).items():
        if not isinstance(m, dict): continue
        if m.get('value') is None: continue
        if atom in defined: candidates.append(atom)
    # 2. Same-family atom pairs (X_now ↔ X_inthehistory; stem ↔ qualifier)
    family_groups = defaultdict(list)
    for atom, m in (av or {}).items():
        if not isinstance(m, dict) or m.get('value') is None: continue
        fam = re.sub(r'(@@\w+)?(_now|_inthehistory|_inthepast\w*)?$', '', atom)
        family_groups[fam].append(atom)
    for fam, atoms in family_groups.items():
        if len(atoms) >= 2:
            for a in atoms:
                if a not in candidates: candidates.append(a)
    # Limit
    candidates = candidates[:25]
    aux_assertions = [ln.strip() for ln in prog_lines
                       if AUX_RE.search(ln) and len(ln) < 400]
    program_text = '\n'.join(aux_assertions[:10]) if aux_assertions else (
        '(no AUXILIARY tags in program; family conflicts only)' if not has_aux else '')
    return candidates, program_text


def process_side(prog_lines, av, sat_like, chart, cache_path):
    """Phase-1 only: no Z3. If sat_like is False, run heuristic candidate
    detection and call arbiter. Cache overrides for downstream eval."""
    if sat_like is not False: return None, None  # only process UNSAT
    if cache_path.exists():
        try: return None, json.loads(cache_path.read_text())
        except: pass
    candidates, program_text = heuristic_candidates(prog_lines, av)
    if not candidates:
        cache_path.write_text(json.dumps({'overrides': [], 'reason': 'no-candidates'}))
        return None, None
    atoms_with_values = []
    for a in candidates:
        m = (av or {}).get(a) or {}
        atoms_with_values.append({
            'atom': a, 'value': m.get('value'),
            'evidence': m.get('evidence'), 'assessment': m.get('assessment'),
        })
    overrides = call_arbiter(chart, atoms_with_values, program_text)
    cache_path.write_text(json.dumps({
        'reason': 'heuristic_candidates', 'overrides': overrides,
        'candidate_atoms': candidates,
    }, indent=2))
    return None, {'overrides': overrides}


def process_pair(pair, vs, charts, cache_dir):
    chart = charts.get(pair.split('__')[0], '')
    elig = False
    arbiter_invocations = 0
    for key, (full_tid, vd) in vs.items():
        inc_raw = vd.get('inclusion',{}).get('raw',{}) or {}
        exc_raw = vd.get('exclusion',{}).get('raw',{}) or {}
        inc_prog = inc_raw.get('smt_program_lines') or []
        exc_prog = exc_raw.get('smt_program_lines') or []
        inc_av = inc_raw.get('patient_var_values_rich') or {}
        exc_av = exc_raw.get('patient_var_values_rich') or {}
        oi = vd.get('inclusion',{}).get('sat_like'); oe = vd.get('exclusion',{}).get('sat_like')
        inc_cache = cache_dir / f'{pair}__{full_tid}__inclusion.json'
        exc_cache = cache_dir / f'{pair}__{full_tid}__exclusion.json'
        ni, ic = process_side(inc_prog, inc_av, oi, chart, inc_cache)
        ne, ec = process_side(exc_prog, exc_av, oe, chart, exc_cache)
        if ic and (ic.get('overrides') or []): arbiter_invocations += 1
        if ec and (ec.get('overrides') or []): arbiter_invocations += 1
        if ni and ne: elig = True
    return pair, elig, arbiter_invocations


# ---------- driver ----------

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


def load_charts():
    out = {}
    for line in open(ROOT/'dataset/clinical_trial/sigir/queries.jsonl'):
        try: o = json.loads(line); out[o['_id']] = o.get('text','')
        except: pass
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
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    if not (ENDPOINT and KEY):
        print('ERROR: need OPENAI_ENDPOINT and OPENAI_API_KEY in env', file=sys.stderr); sys.exit(1)

    cache_dir = pathlib.Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    mine = load_v9(args.mine_dir); charts = load_charts(); gold = load_gold()
    pairs = sorted([(p, vs) for p, vs in mine.items() if p in gold])
    print(f'pairs to process: {len(pairs)}', flush=True)

    verdicts = {}; total_inv = 0; done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_pair, p, vs, charts, cache_dir) for p, vs in pairs]
        for f in as_completed(futs):
            try:
                p, elig, invs = f.result()
            except Exception as e:
                done += 1; print(f'  worker err: {e}', flush=True); continue
            verdicts[p] = elig; total_inv += invs; done += 1
            if done % 50 == 0 or done == len(pairs):
                print(f'  [{done}/{len(pairs)}] arbiter invocations so far: {total_inv}', flush=True)

    print(f'\nPhase 1 done. Arbiter invocations cached: {total_inv}')
    print(f'Cache: {cache_dir}')
    print('Run Phase-2 evaluator (subprocess-isolated Z3) separately to compute verdicts.')


if __name__ == '__main__':
    main()
