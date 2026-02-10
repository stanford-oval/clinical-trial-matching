#!/usr/bin/env python3
"""Run each baseline system's structured output through the unified
_freeform_rationale.prompt template, so all systems produce free-form
clinical prose with the same structure for fair pairwise judging.

Systems:
  v5_verbose  : V5_VERBOSE_V2 (already in v5_verbose_v2_freeform.jsonl)
  trialgpt    : TG corrected (per-criterion structured rationale)
  shah        : Shah lab (per-assessment list with is_met/confidence)
  v5          : Single-shot V5 (sentence-level explanation)

AEGIS keeps its native verbalizer rationale (already free-form clinical prose
from verbalize_aegis_v9_arbiter.prompt).

Outputs:
  matchers/systems/single_shot_llm/v5_freeform.jsonl
  backup/overnight/trialgpt_freeform.jsonl
  backup/overnight/shahlab_freeform.jsonl
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENDPOINT_FULL = os.environ['OPENAI_ENDPOINT']
KEY      = os.environ['OPENAI_API_KEY']
_BASE_M = re.match(r'(https://[^/]+)/openai/deployments/[^/]+', ENDPOINT_FULL)
BASE = _BASE_M.group(1) if _BASE_M else ENDPOINT_FULL

PROMPT_TEMPLATE = (ROOT/'verbalizer/prompts/_freeform_rationale.prompt').read_text()

# Will be set in main() based on --model
_MODEL = 'gpt-4.1'


def _llm(prompt, max_tokens=1500):
    body_d = {
        'messages': [{'role': 'user', 'content': prompt}],
        'response_format': {'type': 'json_object'},
    }
    if _MODEL.startswith('gpt-5') or _MODEL.startswith('o'):
        body_d['max_completion_tokens'] = max_tokens
    else:
        body_d['max_tokens'] = max_tokens
        body_d['temperature'] = 0
    body = json.dumps(body_d).encode()
    url = f"{BASE}/openai/deployments/{_MODEL}/chat/completions?api-version=2024-08-01-preview"
    req = urllib.request.Request(
        url, data=body, headers={'api-key': KEY, 'Content-Type': 'application/json'},
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
        nct = o.get('_id') or o.get('id'); text = o.get('text','') or ''
        lo = text.lower()
        i = lo.find('inclusion criteria'); e = lo.find('exclusion criteria')
        inc = text[i:e] if i>=0 and e>i else (text[i:] if i>=0 else '')
        exc = text[e:] if e>=0 else ''
        out[nct] = (inc, exc)
    return out


# === Per-system adapters: produce a "structured artifact" block from native output ===

def adapt_v5(o):
    """Single-shot V5: explanation is a short prose sentence (or comprehensive)."""
    return o.get('explanation','') or '(no explanation provided)'

def adapt_trialgpt(o):
    """TrialGPT: rationale is already a per-criterion narrative."""
    return o.get('rationale','') or '(no rationale provided)'

def adapt_shah(o):
    parts = ['Per-criterion assessments:']
    for a in (o.get('assessments') or []):
        parts.append(f'- {(a.get("criterion") or "")[:200]}')
        parts.append(f'    is_met={a.get("is_met")}, conf={a.get("confidence")}')
        parts.append(f'    matcher rationale: {(a.get("rationale") or "")[:300]}')
    return '\n'.join(parts)

def adapt_aegis(o):
    """AEGIS: surface SMT statuses + arbiter info + native rationale as the
    structured artifact. The free-form verbalizer will re-render this in
    the same template as the other systems."""
    parts = []
    parts.append(f'Verdict: {o.get("eligibility")}')
    parts.append(f'Inclusion side: {o.get("inclusion_status")}')
    parts.append(f'Exclusion side: {o.get("exclusion_status")}')
    if o.get('arbiter_applied'):
        parts.append(f'Arbiter intervened: yes')
        if o.get('arbiter_overrides'):
            parts.append(f'Arbiter overrides: {json.dumps(o["arbiter_overrides"])[:600]}')
    parts.append('')
    parts.append('AEGIS native verbalizer rationale (this is the matcher\'s prose explanation):')
    parts.append((o.get('rationale','') or '')[:2500])
    return '\n'.join(parts)


SYSTEMS = {
    'v5': {
        'in':      'matchers/systems/single_shot_llm/verdicts.jsonl',
        'out':     'matchers/systems/single_shot_llm/v5_freeform.jsonl',
        'adapter': adapt_v5,
    },
    'trialgpt': {
        'in':      'backup/overnight/trialgpt_corrected.jsonl',
        'out':     'backup/overnight/trialgpt_freeform.jsonl',
        'adapter': adapt_trialgpt,
    },
    'shah': {
        'in':      'backup/overnight/stanford_som_shahlab.jsonl',
        'out':     'backup/overnight/shahlab_freeform.jsonl',
        'adapter': adapt_shah,
    },
    'aegis': {
        'in':      'matchers/systems/aegis/rationales_v9_arbiter.jsonl',
        'out':     'matchers/systems/aegis/aegis_freeform.jsonl',
        'adapter': adapt_aegis,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True, choices=list(SYSTEMS.keys()))
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--model', default='gpt-4.1')
    ap.add_argument('--in', dest='in_path', default=None, help='override source structured-output path')
    ap.add_argument('--out', default=None, help='override output path')
    args = ap.parse_args()
    global _MODEL
    _MODEL = args.model

    cfg = SYSTEMS[args.system]
    src_p = pathlib.Path(args.in_path) if args.in_path else ROOT/cfg['in']
    out_p = pathlib.Path(args.out) if args.out else ROOT/cfg['out']
    out_p.parent.mkdir(parents=True, exist_ok=True)
    adapter = cfg['adapter']

    cache = {}
    if out_p.exists():
        for l in out_p.open():
            try: o=json.loads(l); cache[o['pair']] = o
            except: pass

    src = []
    for line in src_p.open():
        try: o = json.loads(line)
        except: continue
        if o.get('error'): continue
        src.append(o)
    if args.limit: src = src[:args.limit]
    todo = [o for o in src if o['pair'] not in cache]
    print(f'[{args.system}] source={len(src)}  cached={len(cache)}  todo={len(todo)}', flush=True)

    charts = load_charts(); trials = load_trials()

    def proc(o):
        pid, nct = o['pair'].split('__', 1)
        chart = charts.get(pid, '')
        inc, exc = trials.get(nct, ('',''))
        if not chart: return {'pair': o['pair'], 'error': 'missing chart'}
        artifacts = adapter(o)
        prompt = (PROMPT_TEMPLATE
                  .replace('{{CHART}}', chart[:3500])
                  .replace('{{INCLUSION}}', inc[:1500])
                  .replace('{{EXCLUSION}}', exc[:1500])
                  .replace('{{VERDICT}}', o.get('eligibility','') or 'unknown')
                  .replace('{{ARTIFACTS}}', artifacts[:5000]))
        try: out = _llm(prompt)
        except Exception as e:
            return {'pair': o['pair'], 'error': str(e)[:200]}
        return {
            'pair': o['pair'],
            'eligibility': o.get('eligibility',''),
            'rationale': (out.get('rationale','') or '')[:5000],
            'system': f'{args.system}_freeform',
        }

    n = 0
    with out_p.open('a') as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(proc, o) for o in todo]
        for f in as_completed(futs):
            try: rec = f.result()
            except Exception as e:
                print(f'  worker err: {e}', flush=True); continue
            fout.write(json.dumps(rec)+'\n'); fout.flush(); n += 1
            if n % 25 == 0 or n == len(todo):
                print(f'  [{n}/{len(todo)}]', flush=True)
    print(f'[{args.system}] Done. → {out_p}')


if __name__ == '__main__':
    main()
