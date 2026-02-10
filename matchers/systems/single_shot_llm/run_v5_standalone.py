#!/usr/bin/env python3
"""Standalone V5 (V5_TWO_STEP) runner — minimal, swappable LLM backbone.

Inputs: pair list (any of v5_freeform.jsonl etc.), SIGIR queries+corpus.
Output: jsonl with {pair, eligibility, explanation, model, variant}.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[3]
PROMPT_PATH = ROOT/'matchers/systems/single_shot_llm/prompts/two_step.prompt'
ENDPOINT_FULL = os.environ['OPENAI_ENDPOINT']
KEY = os.environ['OPENAI_API_KEY']
_BM = re.match(r'(https://[^/]+)/openai/deployments/[^/]+', ENDPOINT_FULL)
BASE = _BM.group(1) if _BM else ENDPOINT_FULL


def llm(prompt, model, max_tokens=4096):
    body_d = {'messages': [{'role':'user','content':prompt}], 'response_format': {'type': 'json_object'}}
    if model.startswith('gpt-5') or model.startswith('o'):
        body_d['max_completion_tokens'] = max_tokens
        re_env = os.environ.get('GPT5_REASONING_EFFORT')
        if re_env: body_d['reasoning_effort'] = re_env
    else:
        body_d['max_tokens'] = max_tokens
        body_d['temperature'] = 0
    url = f"{BASE}/openai/deployments/{model}/chat/completions?api-version=2024-08-01-preview"
    req = urllib.request.Request(url, data=json.dumps(body_d).encode(),
                                 headers={'api-key': KEY, 'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())['choices'][0]['message']['content']


def parse_json(text):
    if not text: return {}
    m = re.search(r'\{[\s\S]*\}', text)
    if not m: return {}
    try: return json.loads(m.group(0))
    except: return {}


def load_charts():
    return {json.loads(l)['_id']: json.loads(l).get('text','') or ''
            for l in (ROOT/'dataset/clinical_trial/sigir/queries.jsonl').open()}

def load_trials():
    out={}
    for l in (ROOT/'dataset/clinical_trial/sigir/corpus.jsonl').open():
        try: o=json.loads(l)
        except: continue
        nct = o.get('_id') or o.get('id'); out[nct] = o.get('text','') or ''
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-4.1')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--pairs-from', default=str(ROOT/'matchers/systems/aegis/aegis_freeform.jsonl'),
                    help='jsonl with {pair: "..."} entries')
    ap.add_argument('--output', required=True)
    ap.add_argument('--prompt', default=None, help='override prompt path (default: two_step.prompt)')
    args = ap.parse_args()

    pairs = []
    seen = set()
    for l in pathlib.Path(args.pairs_from).open():
        try: o=json.loads(l)
        except: continue
        pa = o.get('pair')
        if pa and pa not in seen: seen.add(pa); pairs.append(pa)
    if args.limit: pairs = pairs[:args.limit]

    OUT = pathlib.Path(args.output); OUT.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if OUT.exists():
        for l in OUT.open():
            try: o=json.loads(l); cache[o['pair']] = o
            except: pass
    todo = [p for p in pairs if p not in cache]
    print(f'pairs={len(pairs)} cached={len(cache)} todo={len(todo)} model={args.model} → {OUT}', flush=True)

    prompt_tmpl = (pathlib.Path(args.prompt).read_text() if args.prompt
                   else PROMPT_PATH.read_text())
    charts = load_charts(); trials = load_trials()

    def proc(pair):
        qid, nct = pair.split('__', 1)
        chart = charts.get(qid, '')
        parent = re.sub(r'(?<=NCT\d{8})[a-z]+$', '', nct)
        trial = trials.get(parent, trials.get(nct, ''))
        if not chart or not trial: return {'pair': pair, 'error': 'missing'}
        # Support both placeholder styles: {trial}/{note}/{chart} and {{TRIAL}}/{{CHART}}
        prompt = (prompt_tmpl
                  .replace('{trial}', trial[:6000])
                  .replace('{note}',  chart[:5000])
                  .replace('{chart}', chart[:5000])
                  .replace('{{TRIAL}}', trial[:6000])
                  .replace('{{CHART}}', chart[:5000]))
        try: text = llm(prompt, args.model)
        except Exception as e: return {'pair': pair, 'error': str(e)[:200]}
        o = parse_json(text)
        rec = {
            'pair': pair,
            'eligibility': (o.get('eligibility') or '').lower() or 'unknown',
            'explanation': (o.get('explanation') or '')[:5000],
            'model': args.model, 'variant': 'V5_TWO_STEP',
        }
        # Passthrough optional structured fields (typed-policy schema, etc.)
        for k in ('inclusion','exclusion','aggregation','category_distribution'):
            if k in o: rec[k] = o[k]
        return rec

    n = 0
    with OUT.open('a') as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(proc, p) for p in todo]
        for f in as_completed(futs):
            try: rec = f.result()
            except Exception as e: print(f'  err: {e}', flush=True); continue
            fout.write(json.dumps(rec)+'\n'); fout.flush(); n += 1
            if n % 50 == 0 or n == len(todo): print(f'  [{n}/{len(todo)}]', flush=True)
    print(f'Done. → {OUT}')

if __name__ == '__main__':
    main()
