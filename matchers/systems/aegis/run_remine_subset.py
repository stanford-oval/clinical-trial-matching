#!/usr/bin/env python3
"""Re-mine a specific subset of (patient_id, full_tid) cohort variants.

Same cmsrc subprocess pipeline as run_remine_full_v10.py, but iterates only
over a JSON list of (pid, tid) tuples instead of every pair-variant in v9.
"""
from __future__ import annotations
import os
import sys
import argparse, json, os, pathlib, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(os.environ.get('VERDICT_ROOT',
    pathlib.Path(__file__).resolve().parents[3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out-root', required=True)
    ap.add_argument('--prompt-root', required=True)
    ap.add_argument('--variants-json', required=True, help='JSON list of [pid, tid] pairs')
    args = ap.parse_args()

    variants = json.loads(pathlib.Path(args.variants_json).read_text())
    print(f'cohort variants to mine: {len(variants)}', flush=True)

    out_root = (ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    cmsrc_dir = pathlib.Path(os.environ.get('CMSRC_DIR',
        pathlib.Path(__file__).resolve().parents[3].parent
        / 'TrialGPT-SMT' / 'cmsrc'))
    cmsrc_python = os.environ.get('CMSRC_PYTHON', sys.executable)
    matcher = cmsrc_dir / 'match_patient_to_trial.py'
    prompt_root = (ROOT/args.prompt_root).resolve()
    print(f'Using prompt_root: {prompt_root}', flush=True)

    log = out_root/'remine.log'

    def proc(item):
        patient_id, full_tid = item
        out_dir = out_root / patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f'{full_tid}__full.json'
        if out_file.exists():
            try:
                if 'inclusion' in json.loads(out_file.read_text()):
                    return f'{patient_id}__{full_tid}\tcached'
            except: pass
        cmd = [cmsrc_python, str(matcher), full_tid, patient_id,
               '--out-root', str(out_root),
               '--prompt-root', str(prompt_root),
               '--miner-mode', 'infer',
               '--data-root', '../dataset/clinical_trial',
               '--build-root', '../build',
               '--prompt-map', str(prompt_root/'prompt_out'/'prompt_map.json')]
        try:
            res = subprocess.run(cmd, cwd=str(cmsrc_dir), capture_output=True, text=True, timeout=1200)
            return f'{patient_id}__{full_tid}\t{"ok" if res.returncode == 0 else "fail rc=" + str(res.returncode)}'
        except subprocess.TimeoutExpired:
            return f'{patient_id}__{full_tid}\ttimeout'
        except Exception as e:
            return f'{patient_id}__{full_tid}\terr {e}'

    n_done=0; n_ok=0
    with log.open('a') as logf, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(proc, v) for v in variants]
        for f in as_completed(futs):
            line = f.result()
            logf.write(line + '\n'); logf.flush()
            n_done += 1
            if 'ok' in line or 'cached' in line: n_ok += 1
            if n_done % 10 == 0 or n_done == len(variants):
                print(f'  [{n_done}/{len(variants)}] ok={n_ok}', flush=True)


if __name__ == '__main__':
    main()
