#!/usr/bin/env python3
"""Self-contained V5_TWO_STEP runner with the unified-template explanation.

Reads pair IDs from the legacy V5 output (so we run on exactly the same
corpus), loads chart + trial directly from the SIGIR jsonls, calls
gpt-4.1, and writes the new comprehensive-rationale output to:

  matchers/systems/single_shot_llm/lm_only_V5_TWO_STEP_templated.jsonl

The old `run.py` chains through several legacy import modules; this
script bypasses them.
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[3]
ENDPOINT = os.environ['OPENAI_ENDPOINT']
KEY      = os.environ['OPENAI_API_KEY']

PROMPT = """You are an expert clinician at PRESCREEN. Reason in two stages.

STAGE 1 — Extract relevant chart facts.
List the patient facts most relevant to this trial's criteria. Use only what is documented.

STAGE 2 — Judge eligibility.
Apply the trial's criteria to the extracted facts. Default to FORWARD on chart silence (prescreen-doctrine).

CLINICAL TRIAL:
{trial}

PATIENT VIGNETTE:
{note}

After your two-stage reasoning, output the final JSON:
{{"eligibility": "eligible"|"ineligible", "explanation": "<see explanation requirements below>"}}

# === EXPLANATION REQUIREMENTS (CRITICAL) ===

The `explanation` field MUST surface per-criterion bookkeeping. Do not summarise with phrases like "the patient meets all inclusion criteria" without grounding each part. Specifically:

- For criteria with direct chart evidence: name the criterion and cite the chart fact (e.g., "Age >= 18: met -- chart documents age 47").
- For criteria where the chart is silent: explicitly say so and label the criterion as "unknown / not addressed in chart". Do not gloss over unknowns.
- For ineligible verdicts: name the specific blocker criterion(s) and cite the chart fact that triggers it.
- For eligible verdicts: state which inclusion criteria are confirmed by chart evidence, which are unknown / silent (prescreen-defaulted), and that no exclusion criterion fires.

Length: 6-12 sentences. Concision is good but is NOT a virtue if it hides the per-criterion bookkeeping above.
"""


def _llm(prompt, max_tokens=2500):
    body = json.dumps({
        'messages': [{'role':'user','content':prompt}],
        'max_tokens': max_tokens, 'temperature': 0,
        'response_format': {'type': 'json_object'},
    }).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/chat/completions?api-version=2024-08-01-preview",
        data=body, headers={'api-key': KEY, 'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        text = json.loads(r.read())['choices'][0]['message']['content'] or ''
    m = re.search(r'\{[\s\S]*\}', text)
    if not m: return {'eligibility': '?', 'explanation': 'no_json'}
    try: o = json.loads(m.group(0))
    except: return {'eligibility': '?', 'explanation': 'parse_err'}
    return {'eligibility': o.get('eligibility','?'), 'explanation': (o.get('explanation') or '')[:4000]}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--pairs-from', default=str(ROOT/'backup/overnight/lm_only_V5_TWO_STEP.jsonl'))
    args = ap.parse_args()

    pairs = []
    for line in open(args.pairs_from):
        try: o=json.loads(line); pairs.append(o['pair'])
        except: pass
    if args.limit: pairs = pairs[:args.limit]

    OUT = ROOT/'matchers/systems/single_shot_llm/lm_only_V5_TWO_STEP_templated.jsonl'
    cache = {}
    if OUT.exists():
        for line in OUT.open():
            try: o=json.loads(line); cache[o['pair']] = o
            except: pass
    todo = [p for p in pairs if p not in cache]
    print(f'pairs={len(pairs)}  cached={len(cache)}  todo={len(todo)}', flush=True)

    charts = load_charts(); trials = load_trials()

    def proc(pair):
        pid, nct = pair.split('__', 1)
        chart = charts.get(pid, ''); trial = trials.get(nct, '')
        if not chart or not trial: return {'pair': pair, 'error': 'missing chart/trial'}
        prompt = PROMPT.format(trial=trial[:5000], note=chart[:5000])
        try: r = _llm(prompt)
        except Exception as e: return {'pair': pair, 'error': str(e)[:200]}
        return {'pair': pair, **r, 'variant': 'V5_TWO_STEP_TEMPLATED', 'model': 'gpt-4.1'}

    n = 0
    with OUT.open('a') as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(proc, p) for p in todo]
        for f in as_completed(futs):
            try: rec = f.result()
            except Exception as e:
                print(f'  worker err: {e}', flush=True); continue
            fout.write(json.dumps(rec)+'\n'); fout.flush(); n += 1
            if n % 25 == 0 or n == len(todo):
                print(f'  [{n}/{len(todo)}]', flush=True)
    print(f'Done. → {OUT}')


if __name__ == '__main__':
    main()
