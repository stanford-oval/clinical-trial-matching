#!/usr/bin/env python3
"""V5_TWO_STEP_BLOCKERS — V5 variant that emits structured `blockers` list.

Reads the same diagonal pairs the original V5 ran on, prompts gpt-4.1 with the
V5_TWO_STEP_BLOCKERS prompt, writes JSONL with {pair, eligibility, explanation,
blockers}.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[3]
ENDPOINT = os.environ.get('OPENAI_ENDPOINT','')
KEY = os.environ.get('OPENAI_API_KEY','')

PROMPT = (ROOT/'matchers/systems/single_shot_llm/prompts/V5_TWO_STEP_BLOCKERS.prompt').read_text()
OUT = ROOT/'matchers/systems/single_shot_llm/v5_blockers.jsonl'


def _llm(prompt):
    body = json.dumps({
        'messages': [{'role':'user','content':prompt}],
        'max_tokens': 1500, 'temperature': 0,
        'response_format': {'type': 'json_object'},
    }).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/chat/completions?api-version=2024-08-01-preview",
        data=body, headers={'api-key': KEY, 'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(json.loads(r.read())['choices'][0]['message']['content'])


def load_charts():
    out = {}
    for line in (ROOT/'dataset/clinical_trial/sigir/queries.jsonl').open():
        try: o = json.loads(line); out[o['_id']] = o.get('text','')
        except: pass
    return out


def load_trials():
    out = {}
    for line in (ROOT/'dataset/clinical_trial/sigir/corpus.jsonl').open():
        try: o = json.loads(line); out[o.get('_id') or o.get('id')] = o.get('text','')
        except: pass
    return out


def process(pair, charts, trials):
    pid, nct = pair.split('__', 1)
    chart = charts.get(pid, ''); trial = trials.get(nct, '')
    if not chart or not trial:
        return {'pair': pair, 'error': 'missing chart or trial'}
    prompt = PROMPT.replace('{{CHART}}', chart[:5000]).replace('{{TRIAL}}', trial[:5000])
    try: o = _llm(prompt)
    except Exception as e:
        return {'pair': pair, 'error': str(e)[:200]}
    return {
        'pair': pair,
        'eligibility': (o.get('eligibility') or '').lower() or 'unknown',
        'explanation': o.get('explanation','') or '',
        'blockers': o.get('blockers') or [],
        'supports': o.get('supports') or [],
        'model': 'gpt-4.1', 'variant': 'V5_TWO_STEP_BLOCKERS',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--pairs-from', default=str(ROOT/'backup/overnight/lm_only_V5_TWO_STEP.jsonl'))
    args = ap.parse_args()
    if not (ENDPOINT and KEY): sys.exit('OPENAI_ENDPOINT and OPENAI_API_KEY required')

    pairs = []
    for line in open(args.pairs_from):
        try: o = json.loads(line); pairs.append(o['pair'])
        except: pass
    if args.limit: pairs = pairs[:args.limit]

    cache = {}
    if OUT.exists():
        for line in OUT.open():
            try: o = json.loads(line); cache[o['pair']] = o
            except: pass
    todo = [p for p in pairs if p not in cache]
    print(f'pairs={len(pairs)}  cached={len(cache)}  todo={len(todo)}')

    charts = load_charts(); trials = load_trials()
    n_done = 0
    with OUT.open('a') as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process, p, charts, trials) for p in todo]
        for f in as_completed(futs):
            try: r = f.result()
            except Exception as e:
                print(f'  worker err: {e}', flush=True); continue
            fout.write(json.dumps(r)+'\n'); fout.flush(); n_done += 1
            if n_done % 25 == 0 or n_done == len(todo):
                print(f'  [{n_done}/{len(todo)}]', flush=True)
    print(f'Done. → {OUT}')


if __name__ == '__main__':
    main()
