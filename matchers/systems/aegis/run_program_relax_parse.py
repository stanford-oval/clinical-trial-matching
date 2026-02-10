#!/usr/bin/env python3
"""Semantic-parser pre-pass for direct numeric inequalities in trial programs.

Scans each trial's inclusion + exclusion SMT program for direct `(> var N)` and
`(< var N)` constraints. For each, asks an LLM whether the clinical intent is
boundary-inclusive (rewrite to >=/<= ) or genuinely strict (keep). Frozen per
trial in `experiments/53_v2_full/program_relax_cache/<trial>.json`.

Output schema per trial:
{
  "trial_id_base": "NCT02359643",
  "rewrites": [
    {"req_name": "REQ1_COMPONENT0_PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE",
     "raw": "(> patient_fever_value_recorded_inthehistory_withunit_days 5.0)",
     "var": "patient_fever_value_recorded_inthehistory_withunit_days",
     "raw_op": ">", "raw_threshold": 5.0,
     "interpreted_op": ">=", "interpreted_threshold": 5.0,
     "rationale": "Kawasaki fever ≥5 days is the clinical-intent boundary"},
    ...
  ]
}
"""
import argparse, json, os, pathlib, re, sys, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[3]
ENDPOINT = os.environ.get('OPENAI_ENDPOINT', '')
KEY = os.environ.get('OPENAI_API_KEY', '')

# Match `(> var N)` or `(< var N)` at the top of an assertion.
_INEQ_RE = re.compile(r'\(\s*([><])\s+([A-Za-z_|][\w@|]*)\s+(-?\d+(?:\.\d+)?)\s*\)')
_NAMED_RE = re.compile(r':named\s+([A-Z0-9_]+)')


def collect_inequalities(prog_lines):
    """For each program line containing `:named <REQ>`, find any inequalities."""
    out = []
    for ln in prog_lines:
        nm = _NAMED_RE.search(ln)
        req = nm.group(1) if nm else None
        for m in _INEQ_RE.finditer(ln):
            op_lex, var, val = m.groups()
            out.append({
                'req_name': req, 'raw': m.group(0),
                'var': var.strip('|'),
                'raw_op_lex': op_lex,
                'raw_op': 'gt' if op_lex == '>' else 'lt',
                'raw_threshold': float(val),
            })
    return out


def collect_per_trial(mine_dir):
    by_trial = {}
    for pdir in pathlib.Path(mine_dir).iterdir():
        if not pdir.is_dir() or pdir.name.startswith('_'): continue
        for fp in pdir.glob('*__full.json'):
            try: o = json.loads(fp.read_text())
            except: continue
            tid = fp.name.replace('__full.json','')
            m = re.match(r'^(NCT\d+)([a-z]?)$', tid)
            if not m: continue
            tid_base = m.group(1)
            inc_raw = (o.get('inclusion',{}) or {}).get('raw',{}) or {}
            exc_raw = (o.get('exclusion',{}) or {}).get('raw',{}) or {}
            entry = by_trial.setdefault(tid_base, {
                'title': inc_raw.get('trial_title') or '',
                'inclusion': inc_raw.get('trial_inclusion_criteria') or '',
                'exclusion': inc_raw.get('trial_exclusion_criteria') or '',
                'inequalities': [],
                'seen': set(),
            })
            for prog in [inc_raw.get('smt_program_lines') or [], exc_raw.get('smt_program_lines') or []]:
                for ineq in collect_inequalities(prog):
                    key = (ineq['req_name'], ineq['var'], ineq['raw_op'], ineq['raw_threshold'])
                    if key in entry['seen']: continue
                    entry['seen'].add(key); entry['inequalities'].append(ineq)
    for v in by_trial.values(): v.pop('seen', None)
    return by_trial


PROMPT = """You are a domain-aware semantic parser for clinical-trial eligibility criteria.

Each inequality below was compiled from a trial criterion. Some criteria written
with strict `>` or `<` have a clinically inclusive intent (e.g., "fever >5 days"
in Kawasaki diagnosis is universally interpreted as ≥5 days); others are
genuinely strict cutoffs (e.g., "creatinine >2.0 mg/dL" used as a renal-failure
exclusion safety margin). Your job: read the trial's criteria text and decide
the clinical-intent operator for each inequality.

Output STRICT JSON, no commentary:
{{
  "rewrites": [
    {{"req_name": "<REQ>", "raw": "<raw inequality>",
      "interpreted_op": "gt" | "ge" | "lt" | "le",
      "rationale": "<one short sentence>"}},
    ...
  ]
}}

Rules:
- Default to **boundary-inclusive** (gt→ge, lt→le) when the criterion is a
  population bound (age, fever days, BMI, performance status, lab thresholds).
- Keep **strict** (gt, lt) only when the criterion clinically excludes the
  boundary (e.g., a safety-margin exclusion, "STRICTLY ABOVE", or numerical
  precision that matters clinically).
- If ambiguous, default to boundary-inclusive.

TRIAL TITLE: {title}

INCLUSION CRITERIA:
{inclusion}

EXCLUSION CRITERIA:
{exclusion}

INEQUALITIES TO INTERPRET:
{ineq_block}
"""


def call_llm(title, inclusion, exclusion, ineqs):
    listing = '\n'.join(
        f'- req=`{i["req_name"]}` raw=`{i["raw"]}` (var=`{i["var"]}`, raw_op={i["raw_op"]}, threshold={i["raw_threshold"]})'
        for i in ineqs
    )
    prompt = PROMPT.format(title=title or '(no title)',
                           inclusion=(inclusion or '(none)')[:3000],
                           exclusion=(exclusion or '(none)')[:3000],
                           ineq_block=listing)
    body = json.dumps({
        'messages': [{'role':'user','content':prompt}],
        'max_tokens': 4000, 'temperature': 0,
        'response_format': {'type': 'json_object'},
    }).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/chat/completions?api-version=2024-08-01-preview",
        data=body, headers={'api-key': KEY, 'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    txt = resp['choices'][0]['message']['content']
    return json.loads(txt)


def parse_one(tid_base, entry, cache_dir, force=False):
    cache_path = cache_dir / f'{tid_base}.json'
    if cache_path.exists() and not force:
        return tid_base, 'cached'
    if not entry['inequalities']:
        cache_path.write_text(json.dumps({'trial_id_base': tid_base, 'rewrites': []}, indent=2))
        return tid_base, 'no-ineqs'
    try:
        result = call_llm(entry['title'], entry['inclusion'], entry['exclusion'], entry['inequalities'])
    except Exception as e:
        return tid_base, f'err: {str(e)[:160]}'
    out_rewrites = []
    intr_by_raw = {}
    for r in (result.get('rewrites') or []):
        intr_by_raw[(r.get('req_name'), r.get('raw'))] = r
    for ineq in entry['inequalities']:
        ir = intr_by_raw.get((ineq['req_name'], ineq['raw']), {})
        out_rewrites.append({
            'req_name': ineq['req_name'],
            'raw': ineq['raw'],
            'var': ineq['var'],
            'raw_op': ineq['raw_op'],
            'raw_threshold': ineq['raw_threshold'],
            'interpreted_op': ir.get('interpreted_op') or ineq['raw_op'],
            'rationale': ir.get('rationale') or '',
        })
    cache_path.write_text(json.dumps({'trial_id_base': tid_base, 'rewrites': out_rewrites}, indent=2))
    return tid_base, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mine-dir', default=str(ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v9_full'))
    ap.add_argument('--cache-dir', default=str(ROOT/'experiments/53_v2_full/program_relax_cache'))
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if not (ENDPOINT and KEY):
        print('ERROR: need OPENAI_ENDPOINT and OPENAI_API_KEY', file=sys.stderr); sys.exit(1)

    cache_dir = pathlib.Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    print(f'Scanning {args.mine_dir}...')
    by_trial = collect_per_trial(args.mine_dir)
    n_ineq = sum(len(v['inequalities']) for v in by_trial.values())
    print(f'  trials: {len(by_trial)}, total inequalities: {n_ineq}')

    done = 0; ok = 0; cached = 0; errs = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(parse_one, t, by_trial[t], cache_dir, args.force): t for t in by_trial}
        for f in as_completed(futs):
            tid, status = f.result()
            if status == 'ok': ok += 1
            elif status == 'cached': cached += 1
            elif status.startswith('err'): errs += 1; print(f'  {tid}: {status}', flush=True)
            done += 1
            if done % 25 == 0 or done == len(by_trial):
                print(f'  [{done}/{len(by_trial)}] ok={ok} cached={cached} errs={errs}', flush=True)
    print(f'Done. → {cache_dir}')


if __name__ == '__main__':
    main()
