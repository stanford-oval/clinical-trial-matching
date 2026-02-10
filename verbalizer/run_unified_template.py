#!/usr/bin/env python3
"""Unified template verbalizer.

Takes each system's native structured output (verdict + per-criterion
rationale or atom-level artifacts) and re-narrates it through a single
shared template. Output is destyled, comprehensive, readable.

Systems handled:
  aegis            — atoms + unsat-core + arbiter overrides
  v5_verbose       — per-criterion judgments + aggregation reasoning (V5_VERBOSE_V2)
  trialgpt         — per-criterion structured rationale (corrected)
  shah             — per-assessment list with is_met/confidence

V5 (single-shot) doesn't need this wrapper because the templated
explanation is baked into V5's matcher prompt and produced natively.

Output: matchers/systems/<system>/rationales_templated.jsonl
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENDPOINT = os.environ['OPENAI_ENDPOINT']
KEY      = os.environ['OPENAI_API_KEY']

UNIFIED_TEMPLATE = """# === ROLE ===
You are a careful clinical-trial eligibility explainer. The verdict has
already been decided by an upstream matcher; your job is to produce a
DESTYLED, readable rationale that faithfully NARRATES THE MATCHER'S OWN
PER-CRITERION BOOKKEEPING in normalized form.

# === FIDELITY RULE (CRITICAL — READ CAREFULLY) ===

You are a verbaliser, not a re-judger. You must:

- ONLY narrate what the matcher's artifacts contain. If the matcher's
  artifacts only address some criteria, your rationale only addresses
  those criteria.
- NEVER infer additional per-criterion judgments from the chart. The
  chart is provided so you can paraphrase the matcher's own chart
  citations into readable English, NOT so you can decide criteria the
  matcher did not address.
- If the matcher did not address a criterion, say so explicitly:
  "the matcher did not address criterion X" — do not look at the chart
  and decide whether X is met.
- Do NOT add reasoning, defaults, or doctrines (e.g., prescreen-default,
  closed-world inference) that the matcher's artifacts do not record.
- PRESERVE THE MATCHER'S ERRORS. If the matcher misread the chart,
  reached an inconsistent conclusion, or used a strange interpretation,
  narrate that faithfully — DO NOT silently correct it. You may say
  "the matcher concluded X" even when the chart says Y. The point is
  to report the matcher's reasoning honestly, not to clean it up.
  (The downstream rater will notice errors on its own; that's their
  job, not yours.)
- If the matcher's verdict and its per-criterion bookkeeping conflict,
  narrate both — "the matcher labelled criterion 5 as not_met but
  concluded eligible" — do not paper over the contradiction.

The intent: each system gets the same destyled, readable presentation,
but with EXACTLY the reasoning depth the system itself produced AND
EXACTLY the same correctness profile. Errors, gaps, and contradictions
in the matcher's output should remain visible after templating.

# === FORMAT ===

For each criterion the matcher addressed, render one short clause that
names the criterion, the matcher's verdict on it (met / not met /
unknown / not addressed / blocker), and the matcher's cited chart
evidence in paraphrased form. Group inclusion-side then exclusion-side.

Length: as long as needed to faithfully cover the matcher's bookkeeping.
Do NOT pad. Do NOT enumerate criteria the matcher didn't touch.

Output STRICT JSON, no commentary outside.

# === INPUTS ===

PATIENT CHART:
{chart}

INCLUSION CRITERIA:
{inclusion}

EXCLUSION CRITERIA:
{exclusion}

UPSTREAM MATCHER ARTIFACTS (verdict + per-criterion bookkeeping the matcher already produced):
{artifacts}

# === OUTPUT ===
{{
  "label": "{label}",
  "rationale": "<plain-English per the rules above>"
}}
"""


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
    out = {}
    for line in (ROOT/'dataset/clinical_trial/sigir/queries.jsonl').open():
        try: o=json.loads(line); out[o['_id']] = o.get('text','')
        except: pass
    return out

def load_trials():
    out = {}
    for line in (ROOT/'dataset/clinical_trial/sigir/corpus.jsonl').open():
        try: o = json.loads(line)
        except: continue
        nct = o.get('_id') or o.get('id')
        text = o.get('text','') or ''
        lo = text.lower()
        i_idx = lo.find('inclusion criteria'); e_idx = lo.find('exclusion criteria')
        inc = ''; exc = ''
        if i_idx >= 0 and e_idx > i_idx: inc = text[i_idx:e_idx]; exc = text[e_idx:]
        elif i_idx >= 0: inc = text[i_idx:]
        elif e_idx >= 0: exc = text[e_idx:]
        out[nct] = (text, inc, exc)
    return out


# ============================================================
# Adapters: each takes a record dict from a system's jsonl and
# returns (verdict_str, artifact_block_str) for the template.
# ============================================================

def adapt_v5_verbose(o):
    parts = []
    for c in (o.get('inclusion_criteria') or []):
        t = (c.get('text') or '')[:200]; s = c.get('status',''); r = (c.get('reasoning') or '')[:300]
        parts.append(f'INCLUSION "{t}" -> {s}: {r}')
    for c in (o.get('exclusion_criteria') or []):
        t = (c.get('text') or '')[:200]; s = c.get('status',''); r = (c.get('reasoning') or '')[:300]
        parts.append(f'EXCLUSION "{t}" -> {s}: {r}')
    if o.get('aggregation_reasoning'):
        parts.append(f'AGGREGATION: {o["aggregation_reasoning"]}')
    return (o.get('eligibility','') or '').lower(), '\n'.join(parts)

def adapt_trialgpt(o):
    return (o.get('eligibility','') or '').lower(), (o.get('rationale') or '')

def adapt_shah(o):
    parts = []
    for a in (o.get('assessments') or []):
        crit = (a.get('criterion') or '')[:200]
        is_met = a.get('is_met')
        conf = a.get('confidence','')
        ratl = (a.get('rationale') or '')[:300]
        parts.append(f'{crit}  [is_met={is_met}, conf={conf}]: {ratl}')
    return (o.get('eligibility','') or '').lower(), '\n'.join(parts)

def adapt_aegis(o):
    inc_st = o.get('inclusion_status') or ''
    exc_st = o.get('exclusion_status') or ''
    interp = []
    if 'SAT' in inc_st and 'UNSAT' not in inc_st:
        interp.append("inclusion_status SAT: every trial inclusion criterion was discharged by the SMT solver, either via direct chart-evidence atom values or via prescreen-default on chart-silent atoms.")
    if 'UNSAT' in inc_st:
        interp.append("inclusion_status UNSAT: at least one inclusion criterion fails; the unsat core / MaxSat min-flip set in the native rationale identifies which.")
    if 'SAT' in exc_st and 'UNSAT' not in exc_st:
        interp.append("exclusion_status SAT: no exclusion criterion fired — every exclusion criterion's SMT encoding remained satisfiable given the patient atoms.")
    if 'UNSAT' in exc_st:
        interp.append("exclusion_status UNSAT: an exclusion criterion fired (was triggered by the patient's atom values).")

    render_hint = (
        "RENDER INSTRUCTIONS for this AEGIS rationale: enumerate every distinct "
        "trial criterion from the INCLUSION CRITERIA and EXCLUSION CRITERIA "
        "text explicitly. For each criterion, write a short clinical-prose "
        "clause. Use natural clinical language, not jargon. Do NOT use the "
        "phrase 'prescreen-default' in the output -- substitute equivalents "
        "like 'chart silent / not documented' or 'no documented evidence'.\n"
        "\n"
        "For each criterion, decide which of these applies and write "
        "accordingly:\n"
        "  - DIRECTLY ADDRESSED (native_rationale cites the chart for it): "
        "render as 'criterion text: met / not met / triggers exclusion -- "
        "<paraphrase the chart citation from native rationale, ideally a short "
        "quote>'.\n"
        "  - PARTIALLY ADDRESSED (criterion has multiple sub-clauses, only "
        "some are chart-grounded): say so honestly, e.g., 'criterion text: "
        "patient is X (per chart) but sub-condition Y is not documented'.\n"
        "  - CHART SILENT: render as 'criterion text: not documented in "
        "chart'. Do NOT invent chart facts not in the native_rationale; do "
        "NOT speculate.\n"
        "  - SMT BLOCKER (named in unsat core / MaxSat flip set in the "
        "native_rationale): make this prominent -- 'criterion text: TRIGGERS "
        "EXCLUSION / FAILS INCLUSION because <chart citation>; this is the "
        "decisive blocker'.\n"
        "\n"
        "Style: write in flowing clinical prose. Group by inclusion side, "
        "then exclusion side. Quote chart text liberally where the native "
        "rationale cites it. The output should read like a clinician's note, "
        "not a SQL result. Do NOT reveal AEGIS's internal terminology "
        "(atoms, SMT, prescreen-default) in the rationale -- the reader sees "
        "the criterion text and the chart, not the matcher's internals."
    )

    art = {
        'verdict': o.get('eligibility'),
        'inclusion_status': inc_st,
        'exclusion_status': exc_st,
        'arbiter_applied': o.get('arbiter_applied'),
        'interpretation_of_status_codes': interp,
        'render_instructions_for_verbalizer': render_hint,
        'native_rationale': o.get('rationale','')[:2000],
    }
    return (o.get('eligibility','') or '').lower(), json.dumps(art, indent=1)


SYSTEMS = {
    'v5_verbose': {
        'in':      'matchers/systems/single_shot_llm/v5_verbose_v2.jsonl',
        'out':     'matchers/systems/single_shot_llm/v5_verbose_v2_templated.jsonl',
        'adapter': adapt_v5_verbose,
    },
    'trialgpt': {
        'in':      'backup/overnight/trialgpt_corrected.jsonl',
        'out':     'backup/overnight/trialgpt_corrected_templated.jsonl',
        'adapter': adapt_trialgpt,
    },
    'shah': {
        'in':      'backup/overnight/stanford_som_shahlab.jsonl',
        'out':     'backup/overnight/stanford_som_shahlab_templated.jsonl',
        'adapter': adapt_shah,
    },
    'aegis': {
        'in':      'matchers/systems/aegis/rationales_v9_arbiter.jsonl',
        'out':     'matchers/systems/aegis/rationales_v9_arbiter_templated.jsonl',
        'adapter': adapt_aegis,
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True, choices=list(SYSTEMS.keys()))
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    cfg = SYSTEMS[args.system]
    in_path = ROOT/cfg['in']; out_path = ROOT/cfg['out']
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {}
    if out_path.exists():
        for l in out_path.open():
            try: o=json.loads(l); cache[o['pair']] = o
            except: pass

    src = []
    for line in in_path.open():
        try: o = json.loads(line)
        except: continue
        if o.get('error'): continue
        src.append(o)
    if args.limit: src = src[:args.limit]
    todo = [o for o in src if o['pair'] not in cache]
    print(f'[{args.system}]  source={len(src)}  cached={len(cache)}  todo={len(todo)}', flush=True)

    charts = load_charts(); trials = load_trials()
    adapter = cfg['adapter']

    def proc(o):
        pid, nct = o['pair'].split('__', 1)
        chart = charts.get(pid, ''); _, inc, exc = trials.get(nct, ('','',''))
        if not chart: return {'pair': o['pair'], 'error': 'missing chart'}
        verdict, artifacts = adapter(o)
        prompt = UNIFIED_TEMPLATE.format(
            chart=chart[:3500], inclusion=inc[:1500], exclusion=exc[:1500],
            artifacts=artifacts[:3500], label=verdict)
        try: out = _llm(prompt)
        except Exception as e: return {'pair': o['pair'], 'error': str(e)[:200]}
        return {'pair': o['pair'], 'eligibility': verdict, 'rationale': out.get('rationale','')[:4000],
                'system': args.system}

    n = 0
    with out_path.open('a') as fout, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(proc, o) for o in todo]
        for f in as_completed(futs):
            try: rec = f.result()
            except Exception as e: print(f'  worker err: {e}', flush=True); continue
            fout.write(json.dumps(rec)+'\n'); fout.flush(); n += 1
            if n % 25 == 0 or n == len(todo):
                print(f'  [{n}/{len(todo)}]', flush=True)
    print(f'Done. → {out_path}')


if __name__ == '__main__':
    main()
