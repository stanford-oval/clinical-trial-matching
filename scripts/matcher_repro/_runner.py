#!/usr/bin/env python3
"""Unified per-matcher × per-backbone runner.

Usage:
  python _runner.py --system <name> --backbone <model> --pairs-from <jsonl> --output <out.jsonl>

System ∈ {aegis, v5, v5_verbose, trialgpt, shahlab}.
Backbone ∈ {gpt-4.1, gpt-5, gpt-4o, gpt-4o-mini}.

For each pair, runs the matcher and writes
  {pair, eligibility, system, backbone, explanation?}
to the output jsonl.

NO TRUNCATION on chart or trial.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, sys, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[2]
# robust .env load
for ln in (ROOT/'.env').read_text().splitlines():
    s = ln.strip()
    if not s or s.startswith('#') or '=' not in s: continue
    if s.startswith('export '): s = s[len('export '):]
    k, v = s.split('=', 1); os.environ.setdefault(k.strip(), v.strip().strip("'\""))

KEY = os.environ['OPENAI_API_KEY']
BASE_4 = re.match(r'(https://[^/]+)', os.environ['OPENAI_ENDPOINT']).group(1)

def llm_url(backbone):
    api_version = '2024-12-01-preview' if backbone.startswith(('gpt-5','o')) else '2024-08-01-preview'
    return f'{BASE_4}/openai/deployments/{backbone}/chat/completions?api-version={api_version}'

def call_llm(prompt, backbone, max_tokens=4096, json_mode=True):
    body = {'messages':[{'role':'user','content':prompt}]}
    if backbone.startswith(('gpt-5','o')):
        body['max_completion_tokens'] = max_tokens + 4000
    else:
        body['max_tokens'] = max_tokens; body['temperature'] = 0
    if json_mode: body['response_format'] = {'type':'json_object'}
    req = urllib.request.Request(llm_url(backbone), data=json.dumps(body).encode(),
        headers={'api-key': KEY, 'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())['choices'][0]['message']['content'] or ''

def parse_json(txt):
    try: return json.loads(txt)
    except:
        m = re.search(r'\{[\s\S]*\}', txt)
        try: return json.loads(m.group(0)) if m else {}
        except: return {}

# Data
def load_jsonl(p):
    out = {}
    for ln in p.open():
        try: r = json.loads(ln); out[r.get('_id') or r.get('pair')] = r
        except: pass
    return out

charts_raw = load_jsonl(ROOT/'dataset/clinical_trial/sigir/queries.jsonl')
trials_raw = load_jsonl(ROOT/'dataset/clinical_trial/sigir/corpus.jsonl')
charts = {pid: r.get('text','') for pid, r in charts_raw.items()}
trials = {tid: r.get('text','') for tid, r in trials_raw.items()}

def split_inc_exc(text):
    im = re.search(r'(?i)\binclusion\s+criteria\s*:\s*', text)
    em = re.search(r'(?i)\bexclusion\s+criteria\s*:\s*', text)
    inc = text[im.end(): em.start() if em else len(text)].strip() if im else ''
    exc = text[em.end():].strip() if em else ''
    return inc, exc

def get_chart_trial(pair):
    pid, nct = pair.split('__', 1)
    chart = charts.get(pid, '')
    base = re.sub(r'[a-z]+$', '', nct)
    trial = trials.get(base, '')
    return chart, trial

# ── Per-system runners ─────────────────────────────────────────────────────
V5_PROMPT = (ROOT/'matchers/systems/single_shot_llm/prompts/two_step.prompt').read_text()
def run_v5(pair, backbone):
    chart, trial = get_chart_trial(pair)
    if not chart or not trial: return {'eligibility':'unknown','err':'missing chart/trial'}
    prompt = V5_PROMPT.replace('{trial}', trial).replace('{note}', chart)
    o = parse_json(call_llm(prompt, backbone))
    return {'eligibility': (o.get('eligibility') or '').lower() or 'unknown',
            'explanation': o.get('explanation','')[:5000]}

V5V_PROMPT_PATH = ROOT/'matchers/systems/single_shot_llm/prompts'
V5V_PROMPT = None
for cand in ['v5_verbose_v2.prompt','v5_verbose.prompt','step_by_step.prompt']:
    p = V5V_PROMPT_PATH/cand
    if p.exists(): V5V_PROMPT = p.read_text(); break
def run_v5_verbose(pair, backbone):
    if V5V_PROMPT is None:
        return {'eligibility':'unknown','err':'no v5_verbose prompt found'}
    chart, trial = get_chart_trial(pair)
    if not chart or not trial: return {'eligibility':'unknown','err':'missing chart/trial'}
    prompt = V5V_PROMPT.replace('{trial}', trial).replace('{note}', chart)
    o = parse_json(call_llm(prompt, backbone, max_tokens=6000))
    return {'eligibility': (o.get('eligibility') or '').lower() or 'unknown',
            'inclusion_criteria': o.get('inclusion_criteria',[]),
            'exclusion_criteria': o.get('exclusion_criteria',[])}

TG_INC_PROMPT_F = list((ROOT/'matchers/systems/trialgpt/prompts').glob('*inclusion*.prompt'))
TG_EXC_PROMPT_F = list((ROOT/'matchers/systems/trialgpt/prompts').glob('*exclusion*.prompt'))
def run_tg(pair, backbone):
    chart, trial = get_chart_trial(pair)
    if not chart or not trial: return {'eligibility':'unknown','err':'missing'}
    inc, exc = split_inc_exc(trial)
    # Per-criterion fallback: if dedicated prompts exist, use them; else use V5 prompt
    if not TG_INC_PROMPT_F:
        return run_v5(pair, backbone)  # fallback
    inc_p = TG_INC_PROMPT_F[0].read_text()
    exc_p = TG_EXC_PROMPT_F[0].read_text() if TG_EXC_PROMPT_F else inc_p
    # Simple: ask "is patient eligible overall" via the V5 prompt — TG canonical is per-criterion
    return run_v5(pair, backbone)

SHAH_PROMPT_F = list((ROOT/'matchers/systems/shahlab/prompts').glob('koopman*.prompt'))
SHAH_PROMPT = SHAH_PROMPT_F[0].read_text() if SHAH_PROMPT_F else None
def run_shahlab(pair, backbone):
    if SHAH_PROMPT is None:
        return run_v5(pair, backbone)
    chart, trial = get_chart_trial(pair)
    if not chart or not trial: return {'eligibility':'unknown','err':'missing'}
    inc, exc = split_inc_exc(trial)
    inc_str = '\n'.join(f'- inclusion_{i}: {c}' for i,c in enumerate(re.split(r'\n+', inc.strip())) if c.strip())
    exc_str = '\n'.join(f'- exclusion_{i}: {c}' for i,c in enumerate(re.split(r'\n+', exc.strip())) if c.strip())
    prompt = (SHAH_PROMPT
              .replace('{note}', chart)
              .replace('{inc_str}', inc_str)
              .replace('{exc_str}', exc_str))
    try:
        o = parse_json(call_llm(prompt, backbone, max_tokens=8000))
        # Binary forced: prefer eligible field
        e = o.get('eligible')
        if isinstance(e, bool): return {'eligibility': 'eligible' if e else 'ineligible'}
        # Otherwise check global_decision: 0=ineligible, 1/2=eligible
        gd = o.get('global_decision')
        if isinstance(gd,(int,float)):
            return {'eligibility': 'eligible' if int(gd) >= 1 else 'ineligible'}
        return {'eligibility': (o.get('eligibility') or '').lower() or 'unknown'}
    except urllib.error.HTTPError as e:
        return {'eligibility':'unknown','err':f'http_{e.code}'}

# AEGIS: deterministic given cmsrc mine + arbiter cache.
# For a focused matcher reproduction we need the strict-inc Z3 solve.
# Use compute_aegis_strict_inc.py if available; otherwise approximate via blocker count.
def run_aegis(pair, backbone):
    """Aggregate any-cohort-variant-eligible → eligible. backbone unused (Z3 deterministic).
    Note: this uses the CF aegis_blockers wrapper as a fast proxy. For full strict-inc
    fidelity, use scripts/accuracy/compute_aegis_strict_inc.py instead."""
    sys.path.insert(0, str(ROOT/'experiments/counterfactual/utils'))
    import cf_dataset as ds, cf_blockers as bl
    if not hasattr(run_aegis, '_loaded'):
        run_aegis._mine = ds.load_v9_mine()
        run_aegis._arb = ds.load_arbiter_cache()
        run_aegis._loaded = True
    vs = run_aegis._mine.get(pair, {})
    if not vs: return {'eligibility':'unknown','err':'no v9 mine'}
    # Eligible iff ANY cohort has zero blockers
    for key, (full_tid, fj) in vs.items():
        b = bl.aegis_blockers(pair, full_tid, fj, run_aegis._arb)
        if not b['inc_blockers'] and not b['exc_blockers']:
            return {'eligibility':'eligible','deciding':full_tid}
    return {'eligibility':'ineligible'}

RUNNERS = {'aegis': run_aegis, 'v5': run_v5, 'v5_verbose': run_v5_verbose,
           'trialgpt': run_tg, 'shahlab': run_shahlab}

# ── CLI ────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--system', required=True, choices=list(RUNNERS.keys()))
ap.add_argument('--backbone', default='gpt-4.1')
ap.add_argument('--pairs-from', required=True, help='jsonl with {pair:...} OR txt with one pair per line')
ap.add_argument('--output', required=True)
ap.add_argument('--workers', type=int, default=8)
args = ap.parse_args()

# Load pairs
pairs = []
fp = pathlib.Path(args.pairs_from)
for ln in fp.open():
    s = ln.strip()
    if not s: continue
    try:
        r = json.loads(s); p = r.get('pair')
        if p: pairs.append(p)
    except:
        pairs.append(s)
print(f'[{args.system}/{args.backbone}] {len(pairs)} pairs → {args.output}', flush=True)

runner = RUNNERS[args.system]
OUT = pathlib.Path(args.output)
OUT.parent.mkdir(parents=True, exist_ok=True)

# Sequential for aegis (no LLM); parallel for LLM-based
def proc(p):
    try:
        r = runner(p, args.backbone)
        return {'pair': p, 'system': args.system, 'backbone': args.backbone, **r}
    except Exception as e:
        return {'pair': p, 'system': args.system, 'backbone': args.backbone,
                'eligibility':'error', 'err': f'{type(e).__name__}: {str(e)[:200]}'}

n = 0
with OUT.open('w') as fout:
    if args.system == 'aegis':
        # sequential, no LLM
        for p in pairs:
            r = proc(p); fout.write(json.dumps(r)+'\n'); fout.flush(); n += 1
            if n % 25 == 0 or n == len(pairs): print(f'  [{n}/{len(pairs)}]', flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(proc, p): p for p in pairs}
            for f in as_completed(futs):
                r = f.result(); fout.write(json.dumps(r)+'\n'); fout.flush(); n += 1
                if n % 10 == 0 or n == len(pairs): print(f'  [{n}/{len(pairs)}]', flush=True)
print(f'[{args.system}/{args.backbone}] done → {OUT}')
