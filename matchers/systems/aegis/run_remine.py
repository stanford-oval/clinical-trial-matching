#!/usr/bin/env python3
"""Re-mine all 539 pairs with the current cmsrc-default prompts.

Output: per-pair cmsrc_out/<patient>/<NCT>__full.json with fresh atoms,
SMT programs, and verdicts.
"""
import os
import sys
import argparse, json, os, pathlib, re, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(os.environ.get('VERDICT_ROOT',
    pathlib.Path(__file__).resolve().parents[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-root', default='experiments/53_v2_full/cmsrc_out_REMINED')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    # Discover all cohort-variant trial IDs from v6 outputs
    v6_root = ROOT/'backup/experiments/53_v2_full/cmsrc_out'
    pair_variants = []   # list of (patient_id, full_trial_id_with_cohort_suffix)
    for pdir in sorted(v6_root.iterdir()):
        if not pdir.is_dir(): continue
        for f in sorted(pdir.glob('NCT*__full.json')):
            tid = f.name.replace('__full.json', '')
            pair_variants.append((pdir.name, tid))
    if args.limit: pair_variants = pair_variants[:args.limit]
    print(f'Re-mining {len(pair_variants)} cohort-variant tasks...')

    out_root = (ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cmsrc_dir = pathlib.Path(os.environ.get('CMSRC_DIR',
        pathlib.Path(__file__).resolve().parents[3].parent
        / 'TrialGPT-SMT' / 'cmsrc'))
    cmsrc_python = os.environ.get('CMSRC_PYTHON', sys.executable)
    matcher = cmsrc_dir / 'match_patient_to_trial.py'  # compiles SMT on the fly
    prompt_root = (ROOT/'experiments/53_v2_full/inputs/prompt_root').resolve()

    def proc(item):
        patient_id, full_tid = item
        # Skip if already done
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
        # If a per-side prompt map is present, dispatch inclusion/exclusion to separate prompts.
        prompt_map = prompt_root / 'prompt_out' / 'prompt_map.json'
        if prompt_map.exists():
            cmd += ['--prompt-map', str(prompt_map)]
        # Note: match_patient_to_trial.py has no --no-cache flag.
        # We use a fresh out-root so its prompt cache is empty → effectively no-cache.
        # LLM and TrialGPT judges are off by default (we don't pass --llm-judge / --trialgpt-judge).
        pair_label = f'{patient_id}__{full_tid}'
        try:
            proc_res = subprocess.run(cmd, cwd=str(cmsrc_dir), capture_output=True, text=True, timeout=300)
            if proc_res.returncode != 0:
                return pair_label, f'rc={proc_res.returncode}: {proc_res.stderr[-200:]}'
            return pair_label, 'ok'
        except subprocess.TimeoutExpired:
            return pair_label, 'timeout'
        except Exception as e:
            return pair_label, str(e)[:200]

    done = 0; ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(proc, p) for p in pair_variants]
        log = (out_root/'remine.log').open('a')
        for f in as_completed(futs):
            pair, status = f.result()
            log.write(f'{pair}\t{status}\n'); log.flush()
            if status in ('ok','cached'): ok += 1
            done += 1
            if done % 25 == 0 or done == len(pair_variants):
                print(f'  [{done}/{len(pair_variants)}] ok={ok}', flush=True)
        log.close()
    print(f'Done. ok={ok}/{len(pair_variants)}')


if __name__ == '__main__':
    main()
