#!/usr/bin/env python3
"""Re-mine ALL pairs with the v10 atom-mining prompt to test the
two-sided impact (FN recovery + FP introduction).

Output: experiments/53_v2_full/cmsrc_out_REMINE_v10_full/<patient>/<NCT>__full.json
"""
import os
import sys
import argparse, json, os, pathlib, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(os.environ.get('VERDICT_ROOT',
    pathlib.Path(__file__).resolve().parents[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-root', default='experiments/53_v2_full/cmsrc_out_REMINE_v10_full')
    ap.add_argument('--prompt-root', default='experiments/53_v2_full/inputs/prompt_root',
                    help='which prompt_root variant to use (canonical for 4.1/4o/4o-mini; per-model fork for tuning)')
    args = ap.parse_args()

    # Collect all (patient_id, full_tid) tasks from existing v9 output
    v9_root = ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v9_full'
    pair_variants = []
    for pdir in sorted(v9_root.iterdir()):
        if not pdir.is_dir(): continue
        for f in sorted(pdir.glob('NCT*__full.json')):
            tid = f.name.replace('__full.json', '')
            pair_variants.append((pdir.name, tid))
    print(f'Total cohort-variant tasks: {len(pair_variants)}')

    out_root = (ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cmsrc_dir = pathlib.Path(os.environ.get('CMSRC_DIR',
        pathlib.Path(__file__).resolve().parents[3].parent
        / 'TrialGPT-SMT' / 'cmsrc'))
    cmsrc_python = os.environ.get('CMSRC_PYTHON', sys.executable)
    matcher = cmsrc_dir / 'match_patient_to_trial.py'
    prompt_root = (ROOT/args.prompt_root).resolve()
    print(f'Using prompt_root: {prompt_root}')

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
            res = subprocess.run(cmd, cwd=str(cmsrc_dir), capture_output=True, text=True, timeout=1200)
            if res.returncode != 0:
                return pair_label, f'rc={res.returncode}: {res.stderr[-200:]}'
            return pair_label, 'ok'
        except subprocess.TimeoutExpired:
            return pair_label, 'timeout'
        except Exception as e:
            return pair_label, str(e)[:200]

    log = (out_root/'remine.log').open('a')
    done = ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(proc, p) for p in pair_variants]
        for f in as_completed(futs):
            pair, status = f.result()
            log.write(f'{pair}\t{status}\n'); log.flush()
            done += 1
            if status in ('ok','cached'): ok += 1
            if done % 25 == 0 or done == len(pair_variants):
                print(f'  [{done}/{len(pair_variants)}] ok={ok}', flush=True)
    print(f'Done. ok={ok}/{len(pair_variants)} → {out_root}')


if __name__ == '__main__':
    main()
