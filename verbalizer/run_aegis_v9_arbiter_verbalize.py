#!/usr/bin/env python3
"""Verbalize AEGIS v9 + arbiter decisions into plain-English rationale.

Reads:
  - v9 mine (cmsrc_out_REMINE_v9_full)
  - arbiter cache (per-pair, per-variant overrides)
  - sigir corpus (trial text)

For each pair, picks the "deciding" cohort variant (eligible variant if any
exists, else the variant with the strongest UNSAT core), applies arbiter
overrides if any, then asks an LLM to verbalize a chart-grounded rationale.

Output: <out>/aegis_v9_arbiter_rationale.jsonl
   {pair, eligibility, rationale, deciding_variant, arbiter_applied}
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, sys, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import z3

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/accuracy/scripts'))
from repro_headline import quote_smt, smt_number, thresh_binding_lines

ENDPOINT = os.environ.get('OPENAI_ENDPOINT', '')
KEY      = os.environ.get('OPENAI_API_KEY', '')

PROMPT_PATH = ROOT/'verbalizer/prompts/verbalize_aegis_v9_arbiter.prompt'


# ---------- data loading ----------

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


def load_arbiter_cache(cache_dir):
    out = {}
    for fp in pathlib.Path(cache_dir).glob('*.json'):
        stem = fp.stem; parts = stem.rsplit('__', 2)
        if len(parts) != 3 or parts[2] not in ('inclusion','exclusion'): continue
        try: out[(parts[0], parts[1], parts[2])] = json.loads(fp.read_text())
        except: pass
    return out


def load_corpus():
    out = {}
    for path in [ROOT/'dataset/clinical_trial/sigir/corpus.jsonl',
                 pathlib.Path(os.environ.get("SIGIR_CORPUS",
        pathlib.Path(__file__).resolve().parents[1].parent
        / "TrialGPT-SMT/dataset/clinical_trial/sigir/corpus.jsonl"))]:
        if not path.exists(): continue
        for line in path.open():
            try: o = json.loads(line); out[o.get('_id') or o.get('id')] = o
            except: pass
        if out: break
    return out


def load_charts():
    out = {}
    for line in open(ROOT/'dataset/clinical_trial/sigir/queries.jsonl'):
        try: o = json.loads(line); out[o['_id']] = o.get('text','')
        except: pass
    return out


# ---------- SMT helpers ----------

def maxsat_blockers_with_overrides(prog_lines, av, overrides):
    """Returns (sat:bool|None, blocker_atoms:list[dict]) via MaxSat.

    blocker_atoms is the minimum-cardinality flip set under the LLM-asserted
    atom values (after arbiter overrides). Each item: {atom, current_value,
    target_value, evidence, assessment}.
    """
    if not prog_lines: return None, []
    drop_map = {o['atom']: o['new_value'] for o in (overrides or [])}
    av_with_overrides = {a: dict(m) if isinstance(m, dict) else m for a, m in (av or {}).items()}
    for atom, newv in drop_map.items():
        if atom in av_with_overrides and isinstance(av_with_overrides[atom], dict):
            av_with_overrides[atom]['value'] = newv
    sys.path.insert(0, str(ROOT/'experiments/counterfactual/utils'))
    from cf_maxsat import maxsat_blocker_list
    try:
        ml = maxsat_blocker_list(prog_lines, av_with_overrides)
    except Exception:
        return None, []
    status = ml.get('status', 'unknown')
    if status == 'sat': return True, []
    if status in ('sat_after_drops', 'unsat_irreducible'): return False, ml.get('blockers', [])
    return None, []


def solve_with_overrides(prog_lines, av, overrides):
    """Returns (sat:bool|None, unsat_core:list[str]). No named asserts to avoid
    Z3 abort under parallelism; unsat_core always returns []."""
    if not prog_lines: return None, []
    drop = {o['atom']: o['new_value'] for o in (overrides or [])}
    asserts = []
    declared = set(); thresh_bound = set(); extra_decls = []
    def bind_thresh(atom):
        if atom in thresh_bound or '__THRESH__' not in atom: return
        lines = thresh_binding_lines(atom, prog_lines=prog_lines)
        if lines: extra_decls.extend(lines); thresh_bound.add(atom)
    def ensure_declared(atom):
        if '__THRESH__' in atom or atom in declared: return
        if not any(re.search(rf'\(declare-(const|fun)\s+{re.escape(atom)}\s+', ln) for ln in prog_lines):
            extra_decls.append(f'(declare-const {quote_smt(atom)} Bool)')
        declared.add(atom)
    for atom, m in (av or {}).items():
        if not isinstance(m, dict): continue
        if atom in drop:
            v = drop[atom]
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
            if sl in ('true','false'):
                asserts.append(f'(assert (= {qn} {sl}))')
    try:
        s = z3.Solver()
        s.from_string('\n'.join(prog_lines)+'\n'+'\n'.join(extra_decls)+'\n'+'\n'.join(asserts))
        r = s.check()
        if r == z3.sat: return True, []
        if r == z3.unsat: return False, []
        return None, []
    except Exception:
        return None, []


def apply_arbiter_to_av(av, overrides):
    """Returns a copy of av with overrides applied + an annotation list."""
    out = {a: dict(m) if isinstance(m, dict) else m for a, m in (av or {}).items()}
    annotations = []
    for o in overrides or []:
        a = o['atom']; new_v = o.get('new_value')
        old_v = (out.get(a) or {}).get('value') if isinstance(out.get(a), dict) else None
        if isinstance(out.get(a), dict):
            out[a]['value'] = new_v
            out[a]['evidence'] = (out[a].get('evidence','') or '') + ' [ARBITER]'
        annotations.append({
            'atom': a, 'old': old_v, 'new': new_v, 'rationale': o.get('rationale','')
        })
    return out, annotations


# ---------- per-pair processing ----------

def fmt_pv(av, max_atoms=50):
    out = []
    for atom, m in list((av or {}).items())[:max_atoms]:
        if not isinstance(m, dict): continue
        v = m.get('value')
        if v is None: continue
        ev = (m.get('evidence') or '')[:300]
        why = (m.get('assessment') or '')[:200]
        out.append(f"  {atom} = {v}\n    evidence: \"{ev}\"\n    why: {why}")
    return '\n'.join(out)


def process_variant(full, arb_inc=None, arb_exc=None):
    """Compute a single variant's verdict + artifacts after arbiter overrides."""
    inc_d = full.get('inclusion') or {}
    exc_d = full.get('exclusion') or {}
    inc_raw = inc_d.get('raw') or {}
    exc_raw = exc_d.get('raw') or {}
    inc_av_orig = inc_raw.get('patient_var_values_rich') or {}
    exc_av_orig = exc_raw.get('patient_var_values_rich') or {}

    inc_av, inc_annot = apply_arbiter_to_av(inc_av_orig, arb_inc)
    exc_av, exc_annot = apply_arbiter_to_av(exc_av_orig, arb_exc)

    # Verdict per side: original cmsrc sat_like, override only if arbiter applied
    oi = inc_d.get('sat_like'); oe = exc_d.get('sat_like')
    inc_core = []; exc_core = []
    inc_status = 'SAT' if oi is True else ('UNSAT' if oi is False else 'unknown')
    exc_status = 'SAT' if oe is True else ('UNSAT' if oe is False else 'unknown')
    # MaxSat: when UNSAT, get the FULL minimum-cardinality flip set (all blockers)
    if oi is False or arb_inc:
        sat, blockers = maxsat_blockers_with_overrides(inc_raw.get('smt_program_lines') or [], inc_av_orig, arb_inc or [])
        if sat is not None:
            oi = sat
            if not sat:
                inc_core = [b['atom'] for b in blockers]
                inc_status = 'UNSAT (MaxSat min flip set)' if not arb_inc else 'UNSAT (after arbiter; MaxSat min flip set)'
            elif arb_inc: inc_status = 'SAT (after arbiter)'
    if oe is False or arb_exc:
        sat, blockers = maxsat_blockers_with_overrides(exc_raw.get('smt_program_lines') or [], exc_av_orig, arb_exc or [])
        if sat is not None:
            oe = sat
            if not sat:
                exc_core = [b['atom'] for b in blockers]
                exc_status = 'UNSAT (MaxSat min flip set)' if not arb_exc else 'UNSAT (after arbiter; MaxSat min flip set)'
            elif arb_exc: exc_status = 'SAT (after arbiter)'
    if oi is False and not inc_core:
        inc_core = (inc_d.get('summary',{}) or {}).get('unsat_core', [])[:30]
    if oe is False and not exc_core:
        exc_core = (exc_d.get('summary',{}) or {}).get('unsat_core', [])[:30]
    elig = (oi is not False) and (oe is not False)
    return {
        'eligible': elig,
        'inc_status': inc_status, 'exc_status': exc_status,
        'inc_core': inc_core, 'exc_core': exc_core,
        'inc_av': inc_av, 'exc_av': exc_av,
        'arb_annotations': inc_annot + exc_annot,
        'inc_raw': inc_raw, 'exc_raw': exc_raw,
    }


def pick_deciding(variants):
    """Pick deciding variant: any eligible, else the one with most-informative core."""
    elig = [v for v in variants if v['eligible']]
    if elig: return elig[0]
    # Prefer one with both sides UNSAT (most informative); tie-break: longer core
    def score(v):
        return (
            (1 if v['inc_status'].startswith('UNSAT') else 0)
          + (1 if v['exc_status'].startswith('UNSAT') else 0),
            len(v.get('inc_core',[])) + len(v.get('exc_core',[])),
        )
    return max(variants, key=score)


def call_llm(prompt):
    body = json.dumps({
        'messages': [{'role':'user','content':prompt}],
        'max_tokens': 1500, 'temperature': 0,
        'response_format': {'type': 'json_object'},
    }).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/chat/completions?api-version=2024-08-01-preview",
        data=body, headers={'api-key': KEY, 'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    txt = resp['choices'][0]['message']['content']
    return json.loads(txt)


def verbalize_one(pair, vs, arbiter_cache, charts, corpus, prompt_template):
    pid, nct_base = pair.split('__', 1)
    chart = charts.get(pid, '')
    # Trial text from sigir corpus
    parent = re.sub(r'(?<=NCT\d{8})[a-z]+$', '', nct_base)
    trial = (corpus.get(parent) or {}).get('text', '') or ''
    # Process each variant + pick deciding
    results = []
    pair_eligible = False
    for key, (full_tid, full) in vs.items():
        arb_inc = arbiter_cache.get((pair, full_tid, 'inclusion'), {}).get('overrides')
        arb_exc = arbiter_cache.get((pair, full_tid, 'exclusion'), {}).get('overrides')
        v = process_variant(full, arb_inc, arb_exc)
        v['full_tid'] = full_tid
        results.append(v)
        if v['eligible']: pair_eligible = True
    deciding = pick_deciding(results)
    label = 'eligible' if pair_eligible else 'ineligible'

    artifacts = {
        'final_label': label,
        'deciding_variant': deciding['full_tid'],
        'inclusion_status': deciding['inc_status'],
        'inclusion_unsat_core': deciding['inc_core'][:20],
        'exclusion_status': deciding['exc_status'],
        'exclusion_unsat_core': deciding['exc_core'][:20],
        'arbiter_invoked': bool(deciding['arb_annotations']),
    }

    arb_text = ''
    if deciding['arb_annotations']:
        arb_text = json.dumps(deciding['arb_annotations'], indent=2)[:1500]
    else:
        arb_text = '(no arbiter overrides applied for this variant)'

    prompt = (prompt_template
              .replace('{{NOTE}}', chart[:3000])
              .replace('{{TRIAL}}', trial[:5000] if trial else '(trial text unavailable)')
              .replace('{{ARTIFACTS}}', json.dumps(artifacts, indent=2)[:2000])
              .replace('{{INC_PV}}', fmt_pv(deciding['inc_av'])[:5000])
              .replace('{{EXC_PV}}', fmt_pv(deciding['exc_av'])[:5000])
              .replace('{{ARBITER_OVERRIDES}}', arb_text))
    try:
        out = call_llm(prompt)
        rationale = (out or {}).get('rationale', '')
    except Exception as e:
        rationale = f'(verbalizer error: {str(e)[:160]})'
    # The verdict is computed from the SMT solver and is NOT decided by the
    # verbalizer LLM. Even when the prompt says "do not relabel", the LLM
    # occasionally returns a label that disagrees with the SMT result; we
    # always trust SMT here.
    verdict_label = label
    return {
        'pair': pair,
        'eligibility': verdict_label,
        'rationale': rationale,
        'deciding_variant': deciding['full_tid'],
        'arbiter_applied': bool(deciding['arb_annotations']),
        'arbiter_overrides': deciding['arb_annotations'],
        'inclusion_status': deciding['inc_status'],
        'exclusion_status': deciding['exc_status'],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mine-dir', default=str(ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v9_full'))
    ap.add_argument('--arbiter-cache', default=str(ROOT/'experiments/53_v2_full/arbiter_cache'))
    ap.add_argument('--out', default=str(ROOT/'matchers/systems/aegis/rationales_v9_arbiter.jsonl'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    if not (ENDPOINT and KEY):
        print('ERROR: need OPENAI_ENDPOINT and OPENAI_API_KEY in env', file=sys.stderr); sys.exit(1)

    print(f'Loading mine + arbiter + corpus...')
    mine = load_v9(args.mine_dir)
    arbiter = load_arbiter_cache(args.arbiter_cache)
    corpus = load_corpus()
    charts = load_charts()
    prompt_template = PROMPT_PATH.read_text()

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if out_path.exists():
        for line in out_path.open():
            try: r = json.loads(line); cache[r['pair']] = r
            except: pass

    pairs = sorted(mine.keys())
    if args.limit: pairs = pairs[:args.limit]
    todo = [p for p in pairs if p not in cache]
    print(f'  pairs total: {len(pairs)}, cached: {len(cache)}, todo: {len(todo)}')

    n_done = 0; n_arb = 0
    with out_path.open('a') as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(verbalize_one, p, mine[p], arbiter, charts, corpus, prompt_template) for p in todo]
        for f in as_completed(futs):
            try: r = f.result()
            except Exception as e:
                print(f'  worker err: {e}', flush=True); continue
            fout.write(json.dumps(r) + '\n'); fout.flush()
            n_done += 1
            if r.get('arbiter_applied'): n_arb += 1
            if n_done % 25 == 0 or n_done == len(todo):
                print(f'  [{n_done}/{len(todo)}] arbiter-applied: {n_arb}', flush=True)
    print(f'Done. Wrote {out_path}')


if __name__ == '__main__':
    main()
