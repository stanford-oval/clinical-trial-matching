#!/usr/bin/env python3
"""V5_VERBOSE_AGG_V2 — same as V5_VERBOSE_AGG but with unambiguous side-specific
labels (included/not_included for inclusions, excluded/not_excluded for exclusions).

Tests the hypothesis that the logical-aggregation errors in V5_VERBOSE_AGG are
driven by the ambiguity of `met`/`not_met` for exclusion criteria.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[3]
ENDPOINT_FULL = os.environ.get('OPENAI_ENDPOINT','')
KEY = os.environ.get('OPENAI_API_KEY','')
_BM = re.match(r'(https://[^/]+)/openai/deployments/[^/]+', ENDPOINT_FULL)
BASE = _BM.group(1) if _BM else ENDPOINT_FULL

PROMPT = (ROOT/'matchers/systems/single_shot_llm/prompts/V5_VERBOSE_AGG_V2.prompt').read_text()
OUT = ROOT/'matchers/systems/single_shot_llm/v5_verbose_v2.jsonl'
_MODEL = 'gpt-4.1'


def _llm(prompt):
    body_d = {
        'messages': [{'role':'user','content':prompt}],
        'response_format': {'type': 'json_object'},
    }
    if _MODEL.startswith('gpt-5') or _MODEL.startswith('o'):
        body_d['max_completion_tokens'] = 3000
    else:
        body_d['max_tokens'] = 3000
        body_d['temperature'] = 0
    body = json.dumps(body_d).encode()
    url = f"{BASE}/openai/deployments/{_MODEL}/chat/completions?api-version=2024-08-01-preview"
    req = urllib.request.Request(url, data=body,
        headers={'api-key': KEY, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(json.loads(r.read())['choices'][0]['message']['content'])


def load_charts():
    out = {}
    for line in (ROOT/'dataset/clinical_trial/sigir/queries.jsonl').open():
        try: o=json.loads(line); out[o['_id']] = o.get('text','')
        except: pass
    return out

def load_trials():
    out = {}
    for line in (ROOT/'dataset/clinical_trial/sigir/corpus.jsonl').open():
        try: o=json.loads(line); out[o.get('_id') or o.get('id')] = o.get('text','')
        except: pass
    return out


def process(pair, charts, trials):
    pid, nct = pair.split('__', 1)
    chart = charts.get(pid, ''); trial = trials.get(nct, '')
    if not chart or not trial: return {'pair': pair, 'error': 'missing chart or trial'}
    prompt = PROMPT.replace('{{CHART}}', chart[:5000]).replace('{{TRIAL}}', trial[:5000])
    try: o = _llm(prompt)
    except Exception as e:
        return {'pair': pair, 'error': str(e)[:200]}
    return {
        'pair': pair,
        'eligibility': (o.get('eligibility') or '').lower() or 'unknown',
        'inclusion_criteria': o.get('inclusion_criteria') or [],
        'exclusion_criteria': o.get('exclusion_criteria') or [],
        'aggregation_reasoning': o.get('aggregation_reasoning','') or '',
        'model': _MODEL, 'variant': 'V5_VERBOSE_AGG_V2',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--pairs-from', default=str(ROOT/'backup/overnight/lm_only_V5_TWO_STEP.jsonl'))
    ap.add_argument('--model', default='gpt-4.1')
    ap.add_argument('--output', default=None)
    args = ap.parse_args()
    if not (BASE and KEY): sys.exit('OPENAI_ENDPOINT and OPENAI_API_KEY required')
    global _MODEL, OUT
    _MODEL = args.model
    if args.output:
        OUT = pathlib.Path(args.output); OUT.parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    for line in open(args.pairs_from):
        try: o=json.loads(line); pairs.append(o['pair'])
        except: pass
    if args.limit: pairs = pairs[:args.limit]

    cache = {}
    if OUT.exists():
        for line in OUT.open():
            try: o=json.loads(line); cache[o['pair']] = o
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
