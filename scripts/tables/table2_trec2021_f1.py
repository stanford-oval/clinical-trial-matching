#!/usr/bin/env python3
"""Reproduce Table 2: F1 on the 363-pair TREC 2021 test set.

Data provenance (verified against the cluster, 2026-09-03):

    trec2021_mine.jsonl          compiled (patient, trial) pairs
      -> prep_2021_split.py      deterministic 80/20 split by TRIAL
                                 ("sort unique NCTs, every 5th NCT -> TEST")
      -> test_tagged.jsonl       436 held-out pairs
      -> parse-clean             drops 73 pairs that do not yield a usable
                                 program ("parse-limited")
      -> test_clean_tagged.jsonl 363 pairs, 190 eligible / 173 ineligible
                                 == the frozen test set every system is scored on

Train and test share no NCT, so there is no trial leakage.

Inputs live in the svpo-rl repo (pulled from
<cluster-host>:<path-to>/svpo-rl). Override with --svpo.

Usage:
    python scripts/tables/table2_trec2021_f1.py
    python scripts/tables/table2_trec2021_f1.py --svpo /path/to/svpo-rl
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys

# Resolve in order: --svpo flag, $SVPO_RL, sibling checkout, then $HOME fallback.
DEFAULT_SVPO = pathlib.Path(
    os.environ.get('SVPO_RL')
    or (pathlib.Path(__file__).resolve().parents[2].parent / 'svpo-rl')
)

# run directory (or data file) -> the Table 2 cell it produces
RUNS = [
    ('GPT-5-mini  VERDICT',        'data/mini_test_rollouts.jsonl', 'pred'),
    ('Qwen2.5-7B  VERDICT',        'eval_runs/base_reeval3072/rollouts.jsonl', 'verdict'),
    ('Qwen2.5-7B  CoT (end-to-end)','eval_runs/qwen_direct/rollouts.jsonl', 'verdict'),
    ('Qwen2.5-7B  VERDICT+distill','eval_runs/fullsup2/rollouts.jsonl', 'verdict'),
    ('Qwen2.5-7B  oracle reader',  'eval_runs/oracle1/rollouts.jsonl', 'verdict'),
]


def load_jsonl(fp: pathlib.Path) -> list:
    out = []
    for line in fp.open():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass          # a few runs have a trailing partial write
    return out


def pair_key(p):
    return tuple(p) if isinstance(p, list) else tuple(p.split('__'))


def score(records, gold, verdict_field):
    """P/R/F1/Acc with 'eligible' as the positive class."""
    tp = fp = fn = tn = 0
    for r in records:
        g = gold.get(pair_key(r['pair']))
        if g is None:
            continue
        gv = (g == 'eligible')
        pv = str(r.get(verdict_field, '')).lower() == 'eligible'
        if pv and gv:     tp += 1
        elif pv:          fp += 1
        elif gv:          fn += 1
        else:             tn += 1
    n = tp + fp + fn + tn
    P = tp / max(1, tp + fp)
    R = tp / max(1, tp + fn)
    F1 = 2 * P * R / max(1e-9, P + R)
    return dict(n=n, P=P, R=R, F1=F1, Acc=(tp + tn) / max(1, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--svpo', type=pathlib.Path, default=DEFAULT_SVPO,
                    help='path to the svpo-rl checkout')
    args = ap.parse_args()

    clean = args.svpo / 'data/test_clean_tagged.jsonl'
    if not clean.exists():
        sys.exit(f'missing frozen test set: {clean}\n'
                 f'pull it with:\n'
                 f'  ssh <cluster-host> "tar czf - -C <path-to>/svpo-rl '
                 f'data/test_clean_tagged.jsonl" | tar xzf - -C {args.svpo}')

    gold = {pair_key(r['pair']): str(r.get('gold', '')).lower()
            for r in load_jsonl(clean)}
    n_e = sum(1 for v in gold.values() if v == 'eligible')
    print(f'frozen test set: {len(gold)} pairs  '
          f'({n_e} eligible / {len(gold) - n_e} ineligible)\n')

    print(f'{"system":<30s}{"n":>5s}{"P":>8s}{"R":>8s}{"F1":>8s}{"Acc":>8s}')
    print('-' * 67)
    for label, rel, field in RUNS:
        fp_ = args.svpo / rel
        if not fp_.exists():
            print(f'{label:<30s}  MISSING: {rel}')
            continue
        m = score(load_jsonl(fp_), gold, field)
        print(f'{label:<30s}{m["n"]:>5d}{m["P"]:>8.3f}{m["R"]:>8.3f}'
              f'{m["F1"]:>8.3f}{m["Acc"]:>8.3f}')

    print('\nPaper Table 2 / \\S6 reference values:')
    print('  GPT-5-mini VERDICT        F1 0.828   Acc 0.838')
    print('  Qwen2.5-7B VERDICT        F1 0.738   Acc 0.697')
    print('  Qwen2.5-7B CoT            F1 0.663')
    print('  Qwen2.5-7B VERDICT+dist.  F1 0.829')
    print('\nNot reproducible from this checkout (separate API runs, outputs not'
          '\nretained): Claude Haiku 4.5 rows, ZSPM, and Xu et al. '
          '(data/xu_assign.jsonl\ncontains atom assignments only and needs a solver pass).')


if __name__ == '__main__':
    main()
