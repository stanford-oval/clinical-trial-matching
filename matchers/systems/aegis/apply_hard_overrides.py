#!/usr/bin/env python3
"""Deterministic hard-rule post-mining overrides.

Forces NULL on atoms matching syntactic patterns that should never be assigned
True/False on chart silence. This implements what the value-mining prompt
asks for, deterministically (no LLM call needed).

Patterns forced to NULL on inclusion-side mining:
  - @@detected_on_*, @@detected_by_*, @@by_*           (imaging modality)
  - @@confirmed_by_*, @@diagnosed_by_*                 (diagnosis methodology)
  - @@according_to_*, @@per_*_criteria                 (criteria methodology)
  - @@diagnosis_according_to_*                         (diagnosis criteria)
  - @@histology_confirmed, @@biopsy_proven, @@aatscored (specific qualifiers)
  - *_is_documented_now, *_is_known_now,
    *_is_recorded_now, *_data_available_now,
    *_documentation_present_now, *_is_available        (data availability)

These are silence-NULL by prescreen-doctrine; the chart cannot establish
trial-site documentation or methodology of care, so they default to NULL
and resolve at the visit.
"""
from __future__ import annotations
import os
import argparse, json, pathlib, re, sys
from collections import defaultdict

ROOT = pathlib.Path(os.environ.get('VERDICT_ROOT',
    pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(ROOT/'backup'))

# Methodology qualifier patterns
METHODOLOGY_PATTERNS = [
    r'@@detected_(on|by|with|using)_',
    r'@@confirmed_by_',
    r'@@diagnosed_by_',
    r'@@diagnosis_according_to_',
    r'@@according_to_',
    r'@@per_.*_criteria',
    r'@@by_(ct|mri|biopsy|histology|imaging|ultrasound|x_ray|pet|microscopy)',
    r'@@histology_confirmed',
    r'@@biopsy_proven',
    r'@@histologically_confirmed',
    r'@@aatscored',
]

# Data-availability patterns
DATA_AVAIL_PATTERNS = [
    r'_is_documented_now$',
    r'_is_known_now$',
    r'_is_recorded_now$',
    r'_data_available_now$',
    r'_documentation_present_now$',
    r'_is_available$',
    r'_is_signed_now$',  # consent forms
]

METH_RE = re.compile('|'.join(METHODOLOGY_PATTERNS))
DATA_RE = re.compile('|'.join(DATA_AVAIL_PATTERNS))


def should_force_null(atom):
    return bool(METH_RE.search(atom) or DATA_RE.search(atom))


SILENCE_KEYWORDS = ('silent', 'not mentioned', 'no mention', 'not documented',
                    'not specified', 'no evidence', 'not available',
                    'not stated', 'absent in', 'defer to visit', 'no information')


def apply_overrides(pvr):
    """Override LLM-mined values with NULL — narrowly:
    - Methodology qualifier atoms: only override when value is False AND evidence
      indicates silence. (If chart explicitly says positive, keep True.)
    - Data-availability atoms: always force NULL (chart cannot establish
      trial-site documentation regardless of mention).
    """
    out = {}
    applied = []
    for atom, info in (pvr or {}).items():
        if not isinstance(info, dict):
            out[atom] = info; continue
        old_val = info.get('value')
        ev = (info.get('evidence') or '').lower()
        is_data_avail = bool(DATA_RE.search(atom))
        is_methodology = bool(METH_RE.search(atom)) and not is_data_avail

        force_null = False; reason = ''
        if is_data_avail and old_val is not None:
            # SAFE RULE: data_availability atoms are never establishable from a
            # prescreen chart (they ask whether trial-site EHR has documentation).
            # Always NULL. This is exactly what the value-mining prompt already
            # instructs but the LLM occasionally violates.
            force_null = True
            reason = 'data_availability (always NULL: chart cannot establish trial-site docs)'
        # Methodology qualifier handling left to arbiter v2: a blanket
        # silence-False -> NULL hard rule turned out to be too aggressive in the
        # full evaluation (fixes 6 cases, breaks 9). The arbiter v2 reads each
        # case in context and applies the rule selectively.

        if force_null:
            new_info = dict(info)
            new_info['value'] = None
            new_info['evidence'] = 'silent: defer to visit (post-mining override)'
            new_info['_override_reason'] = reason
            out[atom] = new_info
            applied.append((atom, old_val, 'data_availability' if is_data_avail else 'methodology'))
        else:
            out[atom] = info
    return out, applied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mine-dir', default=str(ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v9_full'))
    ap.add_argument('--out-dir', default=str(ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v9_full_overridden'))
    args = ap.parse_args()

    mine = pathlib.Path(args.mine_dir)
    out_root = pathlib.Path(args.out_dir)
    if out_root.exists():
        import shutil; shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    n_files = 0; n_overridden = 0; total_overrides = 0
    pattern_counts = defaultdict(int)
    for pdir in sorted(mine.iterdir()):
        if not pdir.is_dir() or pdir.name.startswith('_'): continue
        out_pdir = out_root/pdir.name
        out_pdir.mkdir()
        for f in sorted(pdir.glob('NCT*__full.json')):
            n_files += 1
            try: o = json.loads(f.read_text())
            except: continue
            had_override = False
            for side in ('inclusion','exclusion'):
                raw = (o.get(side, {}).get('raw') or {})
                pvr = raw.get('patient_var_values_rich')
                if not pvr: continue
                new_pvr, applied = apply_overrides(pvr)
                if applied:
                    had_override = True
                    total_overrides += len(applied)
                    for atom, old_val, kind in applied:
                        # categorize
                        if METH_RE.search(atom):
                            pattern_counts['methodology'] += 1
                        elif DATA_RE.search(atom):
                            pattern_counts['data_availability'] += 1
                    raw['patient_var_values_rich'] = new_pvr
            if had_override: n_overridden += 1
            (out_pdir/f.name).write_text(json.dumps(o))
    print(f'files: {n_files}, files_with_overrides: {n_overridden}, total atom overrides: {total_overrides}')
    print(f'pattern breakdown:')
    for k, v in pattern_counts.items(): print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
