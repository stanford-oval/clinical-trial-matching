#!/usr/bin/env python3
"""Convert Step-by-Step (V5_VERBOSE_V2) structured per-criterion output into
free-form clinical prose using the shared verbalizer template. This makes the
Step-by-Step rationale comparable in form to AEGIS's free-form rationale for
fair pairwise judging.

Output: matchers/systems/single_shot_llm/v5_verbose_v2_freeform.jsonl
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENDPOINT = os.environ['OPENAI_ENDPOINT']
KEY      = os.environ['OPENAI_API_KEY']

PROMPT_TEMPLATE = (ROOT/'verbalizer/prompts/_freeform_rationale.prompt').read_text()


def _llm(prompt, max_tokens=1500):
    body = json.dumps({
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': max_tokens, 'temperature': 0,
        'response_format': {'type': 'json_object'},
    }).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/chat/completions?api-version=2024-08-01-preview",
        data=body, headers={'api-key': KEY, 'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        text = json.loads(r.read())['choices'][0]['message']['content'] or ''
    m = re.search(r'\{[\s\S]*\}', text)
    if not m: return {'rationale': '<no_json>'}
    try: return json.loads(m.group(0))
    except: return {'rationale': '<parse_err>'}


def load_charts():
    return {json.loads(l)['_id']: json.loads(l).get('text','') or ''
            for l in (ROOT/'dataset/clinical_trial/sigir/queries.jsonl').open()}


def load_trials():
    out = {}
    for line in (ROOT/'dataset/clinical_trial/sigir/corpus.jsonl').open():
        try: o = json.loads(line)
        except: continue
        nct = o.get('_id') or o.get('id')
        text = o.get('text','') or ''
        lo = text.lower()
        i = lo.find('inclusion criteria'); e = lo.find('exclusion criteria')
        inc = ''; exc = ''
        if i >= 0 and e > i: inc = text[i:e]; exc = text[e:]
        elif i >= 0: inc = text[i:]
        elif e >= 0: exc = text[e:]
        out[nct] = (inc, exc)
    return out


def render_artifacts(o):
    """Produce a structured-but-readable artifact block from V5_VERBOSE_V2 output."""
    parts = []
    parts.append('Per-criterion judgements:')
    for c in (o.get('inclusion_criteria') or []):
        t = (c.get('text') or '')[:200]
        parts.append(f'- INCLUSION "{t}"')
        parts.append(f'    matcher status: {c.get("status")}')
        parts.append(f'    matcher reasoning: {(c.get("reasoning") or "")[:300]}')
    for c in (o.get('exclusion_criteria') or []):
        t = (c.get('text') or '')[:200]
        parts.append(f'- EXCLUSION "{t}"')
        parts.append(f'    matcher status: {c.get("status")}')
        parts.append(f'    matcher reasoning: {(c.get("reasoning") or "")[:300]}')
    if o.get('aggregation_reasoning'):
        parts.append('')
        parts.append('Aggregation reasoning (how matcher combined the per-criterion judgements):')
        parts.append((o.get('aggregation_reasoning') or '')[:1500])
    return '\n'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--in', dest='in_path', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    in_path = pathlib.Path(args.in_path) if args.in_path else ROOT/'matchers/systems/single_shot_llm/v5_verbose_v2.jsonl'
    src = list(in_path.open())
    out_path = pathlib.Path(args.out) if args.out else ROOT/'matchers/systems/single_shot_llm/v5_verbose_v2_freeform.jsonl'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if out_path.exists():
        for l in out_path.open():
            try: o=json.loads(l); cache[o['pair']] = o
            except: pass

    todo = []
    for line in src:
        try: o = json.loads(line)
        except: continue
        if o.get('error'): continue
        if o['pair'] in cache: continue
        todo.append(o)
    if args.limit: todo = todo[:args.limit]
    print(f'pairs to verbalize: {len(todo)} (cached: {len(cache)})')

    charts = load_charts(); trials = load_trials()

    def proc(o):
        pid, nct = o['pair'].split('__', 1)
        chart = charts.get(pid, '')
        inc, exc = trials.get(nct, ('',''))
        if not chart: return {'pair': o['pair'], 'error': 'missing chart'}
        prompt = (PROMPT_TEMPLATE
                  .replace('{{CHART}}', chart[:3500])
                  .replace('{{INCLUSION}}', inc[:1500])
                  .replace('{{EXCLUSION}}', exc[:1500])
                  .replace('{{VERDICT}}', o.get('eligibility','') or 'unknown')
                  .replace('{{ARTIFACTS}}', render_artifacts(o)[:5000]))
        try: out = _llm(prompt)
        except Exception as e:
            return {'pair': o['pair'], 'error': str(e)[:200]}
        return {
            'pair': o['pair'],
            'eligibility': o.get('eligibility',''),
            'rationale': (out.get('rationale','') or '')[:5000],
            'system': 'V5_VERBOSE_V2_freeform',
        }

    n = 0
    with out_path.open('a') as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(proc, o) for o in todo]
        for f in as_completed(futs):
            try: rec = f.result()
            except Exception as e:
                print(f'  worker err: {e}', flush=True); continue
            fout.write(json.dumps(rec)+'\n'); fout.flush(); n += 1
            if n % 25 == 0 or n == len(todo):
                print(f'  [{n}/{len(todo)}]', flush=True)
    print(f'Done. → {out_path}')


if __name__ == '__main__':
    main()
