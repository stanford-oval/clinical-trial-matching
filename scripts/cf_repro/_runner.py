#!/usr/bin/env python3
"""Unified per-system CF self-faithfulness runner with explicit modifier+validator.

Usage:
  python _runner.py --system <name> --modifier <model> --validator <model> \
                    --pairs-from <jsonl|txt> --output <out.jsonl>

System ∈ {aegis, v5, v5_cot, v5_cot_gpt5, v5_blockers, trialgpt, shahlab}.
Modifier ∈ {gpt-4.1, gpt-5}.
Validator ∈ {gpt-4.1, gpt-5}  (gpt-4.1 = itemized; gpt-5 = v3 simclin).

Per pair × system:
  1. Run matcher (first round)
  2. If ineligible, extract blockers
  3. Generate CF via modifier
  4. Validate CF via validator
  5. Rejudge via matcher

NO TRUNCATION ANYWHERE.
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, re, shutil, subprocess, sys, tempfile, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[2]
for ln in (ROOT/'.env').read_text().splitlines():
    s = ln.strip()
    if not s or s.startswith('#') or '=' not in s: continue
    if s.startswith('export '): s = s[len('export '):]
    k, v = s.split('=', 1); os.environ.setdefault(k.strip(), v.strip().strip("'\""))

ap = argparse.ArgumentParser()
ap.add_argument('--system', required=True,
                choices=['aegis','v5','v5_cot','v5_cot_gpt5','v5_blockers','trialgpt','shahlab'])
ap.add_argument('--modifier', default='gpt-4.1', choices=['gpt-4.1','gpt-5'])
ap.add_argument('--validator', default='gpt-4.1', choices=['gpt-4.1','gpt-5'])
ap.add_argument('--pairs-from', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--workers', type=int, default=4)
args = ap.parse_args()

os.environ['CF_MODIFIER_MODEL'] = args.modifier

sys.path.insert(0, str(ROOT/'experiments/counterfactual/utils'))
sys.path.insert(0, str(ROOT/'experiments/counterfactual/05_self_faithfulness/ablation_full_corpus'))
import cf_dataset as ds
import cf_blockers as bl
import cf_generator as gen
import cf_validator as val_module

# Import the per-system runners + filters from run_full_ablation
# (these are the canonical implementations)
import importlib.util
_DRIVER = importlib.util.spec_from_file_location(
    'run_full_ablation',
    str(ROOT/'experiments/counterfactual/05_self_faithfulness/ablation_full_corpus/run_full_ablation.py'))
# Note: can't trivially exec because driver auto-runs main; instead re-use cf_judge + cf_blockers
from cf_judge import V5_PROMPT, TG_PER_CRIT_PROMPT, _split_trial_criteria

# Charts/trials/mine/arbiter
charts = ds.load_charts()
trials = ds.load_trial_text()
mine = ds.load_v9_mine()
arbiter = ds.load_arbiter_cache()

# Pair loader
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
print(f'[CF/{args.system}/mod={args.modifier}/val={args.validator}] {len(pairs)} pairs → {args.output}', flush=True)

# ── LLM helpers (used for validator gpt-5 v3 simclin) ──────────────────────
KEY = os.environ['OPENAI_API_KEY']
BASE = re.match(r'(https://[^/]+)', os.environ['OPENAI_ENDPOINT']).group(1)
def call_llm(prompt, backbone, max_tokens=3000):
    body = {'messages':[{'role':'user','content':prompt}]}
    if backbone.startswith(('gpt-5','o')):
        body['max_completion_tokens'] = max_tokens + 4000
        api_v = '2024-12-01-preview'
    else:
        body['max_tokens'] = max_tokens; body['temperature']=0
        api_v = '2024-08-01-preview'
    body['response_format'] = {'type':'json_object'}
    url = f'{BASE}/openai/deployments/{backbone}/chat/completions?api-version={api_v}'
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={'api-key': KEY, 'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())['choices'][0]['message']['content'] or ''
def parse_json(t):
    try: return json.loads(t)
    except:
        m = re.search(r'\{[\s\S]*\}', t)
        try: return json.loads(m.group(0)) if m else {}
        except: return {}

# ── Validator selector ─────────────────────────────────────────────────────
V3_SIMCLIN_PROMPT = """You are a senior clinical-trial coordinator at the PRESCREEN stage.

ORIGINAL CHART:
{original_chart}

MODIFIED CHART (counterfactual):
{cf_chart}

INCLUSION CRITERIA:
{inclusion}

EXCLUSION CRITERIA:
{exclusion}

CITED BLOCKERS the modifier was asked to address:
{cited_block}

Return strict JSON:
{{
  "coherent":          true|false,
  "flips_target_atom": true|false,
  "keeps_other_facts": true|false,
  "oracle_should_flip":true|false,
  "explanation":       "<one sentence justification>"
}}
"""
def split_inc_exc(text):
    im = re.search(r'(?i)\binclusion\s+criteria\s*:\s*', text)
    em = re.search(r'(?i)\bexclusion\s+criteria\s*:\s*', text)
    inc = text[im.end(): em.start() if em else len(text)].strip() if im else ''
    exc = text[em.end():].strip() if em else ''
    return inc, exc

def validate_cf(orig, cf, cited, preserve, trial, validator):
    if validator == 'gpt-4.1':
        # itemized validator (from cf_validator)
        return val_module.validate(orig, cf, cited, preserve_facts_text=preserve)
    # gpt-5 v3 simclin
    inc, exc = split_inc_exc(trial)
    prompt = V3_SIMCLIN_PROMPT.format(
        original_chart=orig, cf_chart=cf,
        inclusion=inc, exclusion=exc, cited_block=str(cited))
    for _ in range(3):
        try:
            o = parse_json(call_llm(prompt, 'gpt-5'))
            if 'coherent' in o:
                # Bridge v3 → 'valid' for downstream comparison
                o['valid'] = bool(o.get('coherent') and o.get('flips_target_atom') and o.get('keeps_other_facts'))
                return o
        except Exception:
            pass
    return {'valid': False, 'err': 'v3 simclin failed'}

# ── Per-system first-round matcher ─────────────────────────────────────────
def run_v5(chart, trial):
    prompt = V5_PROMPT.format(chart=chart, trial=trial)
    o = parse_json(call_llm(prompt, 'gpt-4.1'))
    return {'eligibility': (o.get('eligibility') or '').lower() or 'unknown',
            'explanation': o.get('explanation','')}
def run_v5_cot(chart, trial, backbone='gpt-4.1'):
    P = (ROOT/'matchers/systems/single_shot_llm/prompts/V5_COT.prompt').read_text()
    prompt = P.replace('{chart}', chart).replace('{trial}', trial)
    o = parse_json(call_llm(prompt, backbone))
    return {'eligibility': (o.get('eligibility') or '').lower() or 'unknown',
            'reasoning': o.get('reasoning',''),
            'decisive_blockers': o.get('decisive_blockers', []),
            'inclusion_supports': o.get('inclusion_supports', []),
            'deferred_at_visit': o.get('deferred_at_visit', [])}
def run_v5_blockers(chart, trial):
    P = (ROOT/'matchers/systems/single_shot_llm/prompts/V5_TWO_STEP_BLOCKERS.prompt').read_text()
    prompt = P.replace('{{CHART}}', chart).replace('{{TRIAL}}', trial)
    o = parse_json(call_llm(prompt, 'gpt-4.1'))
    return {'eligibility': (o.get('eligibility') or '').lower() or 'unknown',
            'blockers': o.get('blockers', []), 'supports': o.get('supports', []),
            'explanation': o.get('explanation','')}
def run_tg(chart, trial):
    inc, exc = _split_trial_criteria(trial)
    inc_rows = []; exc_rows = []
    inc_fail = exc_fail = False
    for c in inc:
        prompt = TG_PER_CRIT_PROMPT.format(chart=chart, inc_or_exc='inclusion', criterion=c)
        o = parse_json(call_llm(prompt, 'gpt-4.1'))
        lbl = (o.get('label') or '').lower()
        inc_rows.append({'criterion': c, 'label': lbl, 'reasoning': o.get('reasoning','')})
        if lbl == 'not included': inc_fail = True
    for c in exc:
        prompt = TG_PER_CRIT_PROMPT.format(chart=chart, inc_or_exc='exclusion', criterion=c)
        o = parse_json(call_llm(prompt, 'gpt-4.1'))
        lbl = (o.get('label') or '').lower()
        if lbl == 'included': lbl = 'excluded'
        elif lbl == 'not included': lbl = 'not excluded'
        exc_rows.append({'criterion': c, 'label': lbl, 'reasoning': o.get('reasoning','')})
        if lbl == 'excluded': exc_fail = True
    elig = 'eligible' if not (inc_fail or exc_fail) else 'ineligible'
    return {'eligibility': elig, 'inc_rows': inc_rows, 'exc_rows': exc_rows}
def run_shahlab(chart, trial):
    P_F = list((ROOT/'matchers/systems/shahlab/prompts').glob('koopman*.prompt'))
    if not P_F: return {'eligibility':'unknown','err':'no koopman prompt'}
    P = P_F[0].read_text()
    inc, exc = split_inc_exc(trial)
    inc_str = '\n'.join(f'- inclusion_{i}: {c}' for i,c in enumerate(re.split(r'\n+', inc.strip())) if c.strip())
    exc_str = '\n'.join(f'- exclusion_{i}: {c}' for i,c in enumerate(re.split(r'\n+', exc.strip())) if c.strip())
    prompt = P.replace('{note}', chart).replace('{inc_str}', inc_str).replace('{exc_str}', exc_str)
    o = parse_json(call_llm(prompt, 'gpt-4.1'))
    e = o.get('eligible')
    if isinstance(e, bool): elig = 'eligible' if e else 'ineligible'
    else:
        gd = o.get('global_decision')
        elig = 'eligible' if isinstance(gd,(int,float)) and int(gd)>=1 else 'ineligible'
    return {'eligibility': elig, 'assessments': o.get('assessments', [])}

# ── CF pipeline per system ─────────────────────────────────────────────────
def proc_aegis(pair):
    vs = mine.get(pair, {})
    if not vs: return {'pair': pair, 'skipped': 'no v9 mine'}
    for key, (full_tid, fj) in vs.items():
        b = bl.aegis_blockers(pair, full_tid, fj, arbiter)
        if b['inc_blockers'] or b['exc_blockers']:
            deciding = full_tid; blk = b; break
    else:
        return {'pair': pair, 'skipped': 'no blockers'}
    pid, _ = pair.split('__', 1)
    chart = charts.get(pid, '')
    base_nct = re.sub(r'[a-z]+$','', pair.split('__',1)[1])
    trial = trials.get(base_nct, '')
    targets = blk['inc_blockers'] + blk['exc_blockers']
    preserve = (blk.get('inc_preserve',[]) + blk.get('exc_preserve',[]))
    other_atoms = (blk.get('inc_all_atoms',[]) + blk.get('exc_all_atoms',[]))
    cf = gen.generate_cf_from_targets(chart, targets, preserve=preserve, other_atoms=other_atoms)
    val_out = validate_cf(chart, cf, json.dumps(targets, indent=2), json.dumps(preserve), trial, args.validator)
    return {'pair': pair, 'system': 'aegis', 'deciding': deciding,
            'targets': targets, 'preserve_count': len(preserve), 'cf_chart': cf,
            'cf_validation': val_out, 'first_verdict': 'ineligible'}

def proc_llm_system(pair, runner, filter_fn):
    pid, _ = pair.split('__', 1)
    chart = charts.get(pid, '')
    base_nct = re.sub(r'[a-z]+$','', pair.split('__',1)[1])
    trial = trials.get(base_nct, '')
    if not chart or not trial: return {'pair': pair, 'skipped': 'missing chart/trial'}
    out = runner(chart, trial)
    if out.get('eligibility') != 'ineligible':
        return {'pair': pair, 'skipped': f'first-round={out.get("eligibility")}', 'first_round': out}
    rationale, supports = filter_fn(out)
    cf = gen.generate_cf_from_rationale(chart, trial, rationale, supports=supports)
    val_out = validate_cf(chart, cf, rationale, supports, trial, args.validator)
    rejudge = runner(cf, trial)
    return {'pair': pair, 'system': args.system, 'first_round': out,
            'cited_rationale': rationale, 'supports': supports,
            'cf_chart': cf, 'cf_validation': val_out, 'rejudged': rejudge,
            'first_verdict': 'ineligible'}

def filter_v5_cot(out):
    blockers = out.get('decisive_blockers', [])
    if not blockers: return out.get('explanation',''), ''
    lines = []
    for b in blockers:
        lines.append(f"  - [{b.get('side','?')}] {b.get('criterion','')}\n    chart fact: {b.get('chart_fact','')}")
    return 'Patient decisive blockers:\n' + '\n'.join(lines), ''
def filter_v5_blockers(out):
    blockers = out.get('blockers', [])
    rationale = ('Patient blockers:\n' + '\n'.join(
        f"  - [{b.get('side','?')}] {b.get('criterion','')}\n    chart fact: {b.get('fact','')}"
        for b in blockers)) if blockers else out.get('explanation','')
    supports = '\n'.join(f"  - {s.get('fact','')}" for s in out.get('supports',[]))
    return rationale, supports
def filter_tg(out):
    inc_block = [f'  - [not included] {r["reasoning"]}'
                 for r in out.get('inc_rows', []) if (r.get('label') or '').lower() == 'not included']
    exc_block = [f'  - [excluded] {r["reasoning"]}'
                 for r in out.get('exc_rows', []) if (r.get('label') or '').lower() == 'excluded']
    parts = []
    if inc_block: parts.append('Inclusion criteria the patient FAILS:\n' + '\n'.join(inc_block))
    if exc_block: parts.append('Exclusion criteria the patient TRIGGERS:\n' + '\n'.join(exc_block))
    return '\n\n'.join(parts) or '(no TG blockers)', ''
def filter_shah(out):
    inc_b, exc_b = [], []
    for a in out.get('assessments', []):
        crit = str(a.get('criterion','')).lower()
        is_met = a.get('is_met')
        evid = a.get('evidence_status','')
        if is_met is False and 'silent' not in evid and 'defaulted' not in evid:
            line = f"  - [{a.get('confidence','?')}] {a.get('rationale','')}"
            if 'inclusion' in crit: inc_b.append(line)
            elif 'exclusion' in crit: exc_b.append(line)
    parts = []
    if inc_b: parts.append('Inclusion criteria the patient FAILS (deterministic):\n' + '\n'.join(inc_b))
    if exc_b: parts.append('Exclusion criteria the patient TRIGGERS (deterministic):\n' + '\n'.join(exc_b))
    return '\n\n'.join(parts) or '(no deterministic blockers)', ''
def filter_v5(out):
    return out.get('explanation',''), ''

PROC = {
    'aegis': lambda p: proc_aegis(p),
    'v5':           lambda p: proc_llm_system(p, run_v5, filter_v5),
    'v5_cot':       lambda p: proc_llm_system(p, lambda c,t: run_v5_cot(c,t,'gpt-4.1'), filter_v5_cot),
    'v5_cot_gpt5':  lambda p: proc_llm_system(p, lambda c,t: run_v5_cot(c,t,'gpt-5'),   filter_v5_cot),
    'v5_blockers':  lambda p: proc_llm_system(p, run_v5_blockers, filter_v5_blockers),
    'trialgpt':     lambda p: proc_llm_system(p, run_tg, filter_tg),
    'shahlab':      lambda p: proc_llm_system(p, run_shahlab, filter_shah),
}

# Run
def safe(p):
    try: return PROC[args.system](p)
    except Exception as e:
        return {'pair': p, 'err': f'{type(e).__name__}: {str(e)[:200]}'}

OUT = pathlib.Path(args.output)
OUT.parent.mkdir(parents=True, exist_ok=True)
n = 0
with OUT.open('w') as fout:
    # Sequential for stability (cf_generator + thread pool seems unstable on pyenv 3.11.9)
    for p in pairs:
        r = safe(p); r['system']=args.system; r['modifier']=args.modifier; r['validator']=args.validator
        fout.write(json.dumps(r, default=str)+'\n'); fout.flush(); n += 1
        if n % 5 == 0 or n == len(pairs): print(f'  [{n}/{len(pairs)}]', flush=True)
print(f'[CF/{args.system}] done → {OUT}')
