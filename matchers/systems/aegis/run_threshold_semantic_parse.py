#!/usr/bin/env python3
"""Semantic-parser pre-pass for AEGIS threshold atoms.

For every `__THRESH__::var::OP::N` atom in a trial, this script asks an LLM to
interpret the original criterion text and output the clinical-intent operator
and threshold. The output is frozen per-trial in a JSON cache.

Why: trial criteria like "fever >5 days" are written with strict inequalities
but the clinical intent is typically inclusive (fever_days >= 5). The default
SMT compilation preserves the lexical operator and rejects boundary chart
values. This pass lets a domain-aware LLM resolve the operator per-criterion
once per trial, frozen and auditable.

Output schema (per trial):
{
  "trial_id_base": "NCT00337116",
  "atoms": {
    "__THRESH__::patient_age_value_recorded_now_in_years::ge::18": {
      "var": "patient_age_value_recorded_now_in_years",
      "raw_op": "ge", "raw_threshold": 18.0,
      "interpreted_op": "ge", "interpreted_threshold": 18.0,
      "rationale": "Adult age criterion; ge:18 matches clinical intent.",
      "criterion_text": "Adults age 18 years or older"
    },
    ...
  }
}

Cache file: experiments/53_v2_full/threshold_cache/<trial_id_base>.json
"""
import argparse, json, os, pathlib, re, sys, urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[3]
ENDPOINT = os.environ.get('OPENAI_ENDPOINT', '')
KEY = os.environ.get('OPENAI_API_KEY', '')

_THRESH_RE = re.compile(r'^__THRESH__::(.+)::(ge|le|gt|lt|eq|ne)::(-?\d+(?:\.\d+)?)$')


PROMPT_TEMPLATE = """You are a domain-aware semantic parser for clinical-trial eligibility criteria.

Each atom below is a Boolean threshold proposition compiled from a trial criterion.
Trial criteria written with strict inequalities (e.g., "fever >5 days") often have
a clinically inclusive intent (the patient at exactly 5 days qualifies); other
times the inequality is genuinely strict (e.g., "creatinine >2.0 mg/dL" as a
renal-failure cutoff). Your job: read the criterion text, decide the
clinical-intent operator, and output a normalized (operator, threshold) for
each atom.

Output STRICT JSON, no commentary:
{{
  "atoms": {{
    "<atom_name>": {{
      "interpreted_op": "ge" | "le" | "gt" | "lt" | "eq" | "ne",
      "interpreted_threshold": <number>,
      "rationale": "<one sentence explaining the inclusivity choice>"
    }},
    ...
  }}
}}

Rules:
- Default to **boundary-inclusive** (ge / le) when the criterion is a population
  bound (age >18 → ge:18; fever >5 days → ge:5; BMI <30 → le:30).
- Keep **strict** (gt / lt) only when the criterion clinically excludes the
  boundary (e.g., "creatinine STRICTLY ABOVE 2.0 mg/dL", or trial language that
  emphasizes "more than" with safety-margin intent).
- The threshold value should match the criterion. Don't shift it.
- If the criterion text is ambiguous, choose boundary-inclusive.

TRIAL TITLE: {title}

INCLUSION CRITERIA:
{inclusion}

EXCLUSION CRITERIA:
{exclusion}

ATOMS TO INTERPRET:
{atoms_block}
"""


def call_llm(title, inclusion, exclusion, atoms):
    atoms_block = '\n'.join(
        f'- `{a["atom_name"]}` (raw lexical: var=`{a["var"]}`, op=`{a["raw_op"]}`, threshold={a["raw_threshold"]})'
        for a in atoms
    )
    prompt = PROMPT_TEMPLATE.format(
        title=title or '(no title)',
        inclusion=(inclusion or '(none)')[:3000],
        exclusion=(exclusion or '(none)')[:3000],
        atoms_block=atoms_block,
    )
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


def collect_thresh_atoms_by_trial(mine_dir):
    """Scan the mine for unique THRESH atoms per trial. Returns:
    {trial_id_base: {'title':..., 'inclusion':..., 'exclusion':..., 'atoms': [...]}, ...}
    where atoms is a list of {atom_name, var, raw_op, raw_threshold}.
    """
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
                'atoms': {},
            })
            for raw in (inc_raw, exc_raw):
                av = raw.get('patient_var_values_rich') or {}
                for atom in av:
                    if atom in entry['atoms']: continue
                    m2 = _THRESH_RE.match(atom)
                    if not m2: continue
                    var, op, n = m2.groups()
                    entry['atoms'][atom] = {
                        'atom_name': atom, 'var': var, 'raw_op': op, 'raw_threshold': float(n),
                    }
    return by_trial


def parse_one_trial(tid_base, entry, cache_dir, force=False):
    cache_path = cache_dir / f'{tid_base}.json'
    if cache_path.exists() and not force:
        return tid_base, 'cached'
    atoms = list(entry['atoms'].values())
    if not atoms:
        cache_path.write_text(json.dumps({'trial_id_base': tid_base, 'atoms': {}}, indent=2))
        return tid_base, 'no-atoms'
    try:
        result = call_llm(entry['title'], entry['inclusion'], entry['exclusion'], atoms)
    except Exception as e:
        return tid_base, f'err: {str(e)[:160]}'
    out = {'trial_id_base': tid_base, 'atoms': {}}
    interpreted = (result or {}).get('atoms') or {}
    for a in atoms:
        name = a['atom_name']
        ir = interpreted.get(name) or {}
        out['atoms'][name] = {
            **a,
            'interpreted_op': ir.get('interpreted_op') or a['raw_op'],
            'interpreted_threshold': ir.get('interpreted_threshold') if ir.get('interpreted_threshold') is not None else a['raw_threshold'],
            'rationale': ir.get('rationale') or '',
        }
    cache_path.write_text(json.dumps(out, indent=2))
    return tid_base, 'ok'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mine-dir', default=str(ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v9_full'))
    ap.add_argument('--cache-dir', default=str(ROOT/'experiments/53_v2_full/threshold_cache'))
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    if not (ENDPOINT and KEY):
        print('ERROR: need OPENAI_ENDPOINT and OPENAI_API_KEY in env', file=sys.stderr)
        sys.exit(1)

    mine_dir = pathlib.Path(args.mine_dir)
    cache_dir = pathlib.Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f'Scanning {mine_dir} for THRESH atoms...')
    by_trial = collect_thresh_atoms_by_trial(mine_dir)
    trials = sorted(by_trial.keys())
    if args.limit: trials = trials[:args.limit]
    n_atoms = sum(len(by_trial[t]['atoms']) for t in trials)
    print(f'  {len(trials)} trials, {n_atoms} unique THRESH atoms')

    done = 0; ok = 0; cached = 0; errs = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(parse_one_trial, t, by_trial[t], cache_dir, args.force): t for t in trials}
        for f in as_completed(futs):
            tid, status = f.result()
            if status == 'ok': ok += 1
            elif status == 'cached': cached += 1
            elif status == 'no-atoms': pass
            else: errs += 1; print(f'  {tid}: {status}', flush=True)
            done += 1
            if done % 25 == 0 or done == len(trials):
                print(f'  [{done}/{len(trials)}] ok={ok} cached={cached} errs={errs}', flush=True)
    print(f'Done. ok={ok} cached={cached} errs={errs} → {cache_dir}')


if __name__ == '__main__':
    main()
