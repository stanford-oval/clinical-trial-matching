#!/usr/bin/env python3
"""Re-mine ONLY the AEGIS false-negative pairs (against the 5-judge gold)
with the updated mining prompt to test whether the methodology-qualifier
NULL guidance recovers them.

Output: experiments/53_v2_full/cmsrc_out_REMINE_v10_fn/<patient>/<NCT>__full.json
"""
import os
import sys
import argparse, json, os, pathlib, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(os.environ.get('VERDICT_ROOT',
    pathlib.Path(__file__).resolve().parents[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out-root', default='experiments/53_v2_full/cmsrc_out_REMINE_v10_fn')
    args = ap.parse_args()

    # Find FN pairs against the 5-judge gold
    gold = json.load((ROOT/'experiments/accuracy/data/gold_5sys.json').open())['gold']
    aegis = {}
    for line in (ROOT/'matchers/systems/aegis/rationales_v9_arbiter.jsonl').open():
        try: o = json.loads(line)
        except: continue
        if o.get('error'): continue
        e = (o.get('eligibility') or '').lower()
        if e in ('eligible','ineligible'):
            aegis[o['pair']] = (e == 'eligible')
    fns = sorted(p for p in gold if p in aegis and not aegis[p] and gold[p])
    print(f'FN pairs to re-mine: {len(fns)}')

    # Discover all cohort variants for each FN pair (read from existing v9 out)
    v9_root = ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v9_full'
    pair_variants = []
    for pair in fns:
        pid, _ = pair.split('__', 1)
        parent_nct = pair.split('__', 1)[1]
        pdir = v9_root/pid
        if not pdir.exists():
            print(f'  WARN: no v9 dir for {pid}')
            continue
        for f in sorted(pdir.glob(f'{parent_nct}*__full.json')):
            tid = f.name.replace('__full.json', '')
            pair_variants.append((pid, tid))
    print(f'Cohort-variant tasks: {len(pair_variants)}')

    out_root = (ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cmsrc_dir = pathlib.Path(os.environ.get('CMSRC_DIR',
        pathlib.Path(__file__).resolve().parents[3].parent
        / 'TrialGPT-SMT' / 'cmsrc'))
    cmsrc_python = os.environ.get('CMSRC_PYTHON', sys.executable)
    matcher = cmsrc_dir / 'match_patient_to_trial.py'
    prompt_root = (ROOT/'experiments/53_v2_full/inputs/prompt_root').resolve()

    def proc(item):
        patient_id, full_tid = item
        out_dir = out_root / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'{full_tid}__full.json'
        if out_file.exists():
            return f'{patient_id}__{full_tid}', 'cached'
        cmd = [
            cmsrc_python, str(matcher),
            full_tid, patient_id,
            '--out-root', str(out_root),
            '--prompt-root', str(prompt_root),
            '--miner-mode', 'infer',
            '--data-root', '../dataset/clinical_trial',
            '--build-root', '../build',
        ]
        prompt_map = prompt_root / 'prompt_out' / 'prompt_map.json'
        if prompt_map.exists():
            cmd += ['--prompt-map', str(prompt_map)]
        pair_label = f'{patient_id}__{full_tid}'
        try:
            res = subprocess.run(cmd, cwd=str(cmsrc_dir), capture_output=True, text=True, timeout=300)
            if res.returncode != 0:
                return pair_label, f'rc={res.returncode}: {res.stderr[-200:]}'
            return pair_label, 'ok'
        except subprocess.TimeoutExpired:
            return pair_label, 'timeout'
        except Exception as e:
            return pair_label, str(e)[:200]

    done = 0; ok = 0
    log = (out_root/'remine.log').open('a')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(proc, p) for p in pair_variants]
        for f in as_completed(futs):
            pair, status = f.result()
            log.write(f'{pair}\t{status}\n'); log.flush()
            done += 1
            if status in ('ok','cached'): ok += 1
            if done % 5 == 0 or done == len(pair_variants):
                print(f'  [{done}/{len(pair_variants)}] ok={ok}', flush=True)
    print(f'Done. ok={ok}/{len(pair_variants)} → {out_root}')


if __name__ == '__main__':
    main()
