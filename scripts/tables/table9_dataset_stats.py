#!/usr/bin/env python3
"""Reproduce Table 9: Evaluation dataset statistics.

Counts patients, trials, and pairs for each evaluation slice used in the
paper (SIGIR headline, TREC 2021 subset, clinician audit samples).
"""
from __future__ import annotations
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]


def count_lines(fp: pathlib.Path) -> int:
    if not fp.exists(): return 0
    return sum(1 for _ in fp.open())


def load_jsonl(fp: pathlib.Path):
    if not fp.exists(): return []
    return [json.loads(l) for l in fp.open() if l.strip()]


def main():
    rows = []

    # SIGIR raw corpus + queries
    sigir_q = ROOT / 'dataset/clinical_trial/sigir/queries.jsonl'
    sigir_c = ROOT / 'dataset/clinical_trial/sigir/corpus.jsonl'
    if sigir_q.exists() or sigir_c.exists():
        rows.append(('SIGIR raw',   count_lines(sigir_q), count_lines(sigir_c), '—'))
    else:
        # Local-only corpus (see docs/DATA.md). Say so rather than printing 0,
        # which reads like a finding.
        rows.append(('SIGIR raw',   'n/a', 'n/a', 'corpus not present'))

    # SIGIR gold (5-system judge panel)
    gold_fp = ROOT / 'data/gold/gold_5sys_freeform_balanced.json'
    if gold_fp.exists():
        gold = json.loads(gold_fp.read_text())
        # Two schemas seen in the wild: list of {pair, ...} OR {gold: {pair: label}}.
        if isinstance(gold, dict) and 'gold' in gold and isinstance(gold['gold'], dict):
            pair_keys = list(gold['gold'].keys())
        elif isinstance(gold, list):
            pair_keys = [p.get('pair','') for p in gold]
        elif isinstance(gold, dict):
            pair_keys = [p.get('pair','') for p in (gold.get('pairs') or [])]
        else:
            pair_keys = []
        pids = {k.split('__')[0] for k in pair_keys if '__' in k}
        tids = {k.split('__')[1] for k in pair_keys if '__' in k}
        rows.append(('SIGIR gold (headline)', len(pids), len(tids), len(pair_keys)))

    # Clinician audit (32-pair sample)
    audit32 = ROOT / 'experiments/clinician_validation/instrument/samples/sample_32pairs_balanced_seed7.json'
    if audit32.exists():
        s = json.loads(audit32.read_text())
        pairs = s if isinstance(s, list) else s.get('pairs', [])
        rows.append(('Clinician audit (32-pair)', '—', '—', len(pairs)))

    # Clinician CF audit (K=7)
    cf_audit = ROOT / 'experiments/clinician_validation/cf_audit/cf_audit_K7_cell3.json'
    if cf_audit.exists():
        s = json.loads(cf_audit.read_text())
        topics = s.get('topics', s if isinstance(s, list) else [])
        rows.append(('Clinician CF audit (K=7, cell3)', '—', '—', len(topics)))

    # TREC 2021 qwen eval
    trec = ROOT / 'abductive_supervision/data/trec2021_qwen_eval.jsonl'
    if trec.exists():
        n = count_lines(trec)
        recs = load_jsonl(trec)
        pids = {r['pair'].split('__')[0] for r in recs if '__' in r.get('pair','')}
        tids = {r['nct'] for r in recs if r.get('nct')}
        rows.append(('TREC 2021 (qwen eval subset)', len(pids), len(tids), n))

    # Reverse CF samples
    rev = ROOT / 'experiments/counterfactual/06_reverse_cf/out'
    if rev.exists():
        for f in sorted(rev.glob('reverse_cf_aegis_hard*_n*.jsonl')):
            if 'rejudged' in f.name: continue
            n = count_lines(f)
            rows.append((f'Reverse CF: {f.stem}', '—', '—', n))

    # Print
    print(f'{"slice":<40s} {"pts":>5s} {"trials":>7s} {"pairs":>7s}')
    print('-' * 65)
    for slice_name, pts, trials, pairs in rows:
        print(f'{slice_name:<40s} {str(pts):>5s} {str(trials):>7s} {str(pairs):>7s}')


if __name__ == '__main__':
    main()
