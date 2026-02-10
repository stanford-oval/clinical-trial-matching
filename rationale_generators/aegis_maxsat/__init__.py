"""AEGIS rationale generator: MaxSat-minimum atom flip set.

Given AEGIS's mined SMT program + LLM-asserted atom values, finds the
minimum-cardinality set of atom assignments to flip such that the program
becomes SAT. Implementation: Z3 Optimize (Partial MaxSAT) — see
experiments/counterfactual/utils/cf_maxsat.py.
"""
from __future__ import annotations
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/counterfactual/utils'))


def generate(matcher_output: dict, pair_meta: dict) -> dict:
    """matcher_output: AEGIS per-pair record with keys:
       - 'targets': list of {atom, current_value, target_value, evidence, assessment}
                    (already produced by cf_blockers.aegis_blockers; this generator
                    is a pass-through with optional re-resolution of numeric thresholds)
       - 'preserve': list of must-stay-supported atoms
       - 'other_atoms': list of all-other atom snapshots (for silent-qualifier preservation)
    """
    targets = list(matcher_output.get('targets') or [])
    # Resolve any numeric atom whose target_value is null but threshold cache
    # provides a constraint (the AGE bug fix from May-14).
    _resolve_age_targets(targets, pair_meta.get('trial_id', ''))
    return {
        'kind':         'atom_targets',
        'atom_targets': targets,
        'preserve':     matcher_output.get('preserve') or [],
        'other_atoms':  matcher_output.get('other_atoms') or [],
        'source':       'aegis.maxsat',
    }


def _resolve_age_targets(targets: list, trial_id: str) -> None:
    """In-place patch: numeric age atoms with target=None get a concrete in-range value."""
    if not trial_id: return
    import re, json
    base = re.sub(r'[a-z]+$', '', trial_id)
    p = ROOT/'experiments/53_v2_full/threshold_cache'/f'{base}.json'
    if not p.exists(): return
    try: thresh = json.load(p.open()).get('atoms', {})
    except Exception: return
    for t in targets:
        cv = t.get('current_value'); tv = t.get('target_value'); atom = t.get('atom','')
        if not (isinstance(cv, (int,float)) and tv is None
                and '__THRESH__' not in atom and 'age' in atom.lower()):
            continue
        bounds = _bounds_for(atom, thresh)
        concrete = _pick_concrete(bounds)
        if concrete is not None:
            t['target_value'] = concrete

def _bounds_for(atom, thresh):
    lo=hi=eq=None
    for _, ta in thresh.items():
        if ta.get('var') != atom: continue
        op, th = ta.get('interpreted_op'), ta.get('interpreted_threshold')
        if op in ('ge','gt'):
            new = th if op=='ge' else th+1
            lo = new if lo is None else max(lo, new)
        elif op in ('le','lt'):
            new = th if op=='le' else th-1
            hi = new if hi is None else min(hi, new)
        elif op == 'eq': eq = th
    return {'lo': lo, 'hi': hi, 'eq': eq}

def _pick_concrete(b):
    if b['eq'] is not None: return b['eq']
    if b['lo'] is not None and b['hi'] is not None: return int((b['lo']+b['hi'])/2)
    if b['lo'] is not None: return b['lo']+1
    if b['hi'] is not None: return b['hi']-1
    return None
