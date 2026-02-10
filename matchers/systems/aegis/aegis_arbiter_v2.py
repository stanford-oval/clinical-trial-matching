#!/usr/bin/env python3
"""AEGIS Phase-2 arbiter — uses Z3 unsat-core (computed at mine time) +
syntactic classification of every unsat-core item.

Classification:
  - GENUINE: every non-atom item in the unsat core matches
    `REQ<N>_COMPONENT<M>_OTHER_REQUIREMENTS` — these are the trial's
    stated criteria. Patient atoms in the core are the real blockers.
    Verdict ineligible stands.

  - MINING_SELF_CONTRADICTION: the unsat core contains at least one
    assertion that's NOT a real criterion — typically:
      * `AUTO_AUXILIARY_QUAL_IMPLIES_STEM_*` (auto-generated qualifier-stem
        implications, often wrong polarity for "absent_*"-style qualifiers)
      * `REQ<N>_AUXILIARY<M>` (program-defined auxiliary)
      * `REQ<N>_COMPONENT<M>_PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE`
        (strict-evidence assertion that violates prescreen-doctrine)
      * `REQ<N>_COMPONENT<M>_DATA_AVAILABILITY` (logistical)
    The contradiction is in the mining/encoding, not in the patient-vs-trial
    relationship. Hand to LLM resolver: given trial text, SMT program,
    unsat core, and mined atom values, propose atom-value overrides that
    resolve the contradiction faithfully.

Per-pair, per-side:
  - Read v10 mine output (inclusion side) / v9 mine output (exclusion side).
  - If status == unsat: classify; if mining_self_contradiction, call LLM.
  - Cache result to experiments/53_v2_full/arbiter_v2_cache/<pair>__<variant>__<side>.json
  - Apply overrides and re-solve to compute the patched verdict.

Invocation: python matchers/systems/aegis/aegis_arbiter_v2.py
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[3]
ENDPOINT = os.environ.get('OPENAI_ENDPOINT','')
KEY      = os.environ.get('OPENAI_API_KEY','')

V9   = ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v9_full'
V10  = ROOT/'experiments/53_v2_full/cmsrc_out_REMINE_v10_full'
CACHE = ROOT/'experiments/53_v2_full/arbiter_v2_cache'

# Real trial criteria — any assertion of the form REQ<N>_COMPONENT<M>_<suffix>
# regardless of the suffix (OTHER_REQUIREMENTS, PRESCREEN_NOTES_MUST_COMPLETELY_SUFFICE,
# DATA_AVAILABILITY, etc.). The suffix encodes the criterion's semantic class but the
# REQ_COMPONENT prefix marks it as a stated trial requirement.
REAL_REQ_PATTERN = re.compile(r'^REQ\d+_COMPONENT\d+_')


def classify_unsat_core(core, side):
    """Returns ('genuine'|'mining_self_contradiction', list of non-real assertions)."""
    non_real_structural = []
    patient_atoms = []
    real_reqs = []
    for item in (core or []):
        if item.startswith('patient_'):
            patient_atoms.append(item)
        elif REAL_REQ_PATTERN.match(item):
            real_reqs.append(item)
        else:
            non_real_structural.append(item)
    if non_real_structural:
        return 'mining_self_contradiction', {
            'non_real_structural': non_real_structural,
            'patient_atoms_in_core': patient_atoms,
            'real_reqs_in_core': real_reqs,
        }
    return 'genuine', {'patient_atoms_in_core': patient_atoms,
                      'real_reqs_in_core': real_reqs}


ARBITER_PROMPT = """You are auditing an SMT-program-level contradiction that arose during clinical-trial eligibility matching. Z3 returned UNSAT for the inclusion side. Your job is to decide whether this UNSAT is GENUINE (patient really fails the criterion) or a MINING ARTIFACT (a compiler-emitted helper assertion is wrongly forcing values on chart-silent atoms).

# === DEFAULT POSITION: GENUINE ===

By default, an UNSAT means the patient is INELIGIBLE for this cohort variant. This is the correct conclusion most of the time. **You must require POSITIVE EVIDENCE that the contradiction is a mining artifact before proposing any override.** When in doubt, classify as `genuine_after_review` and return empty overrides.

A patient is GENUINELY INELIGIBLE — return NO overrides — when:
- The chart describes findings that match the trial's exclusion or fail an inclusion (e.g., trial wants stage IV NSCLC; chart describes stage I lung adenocarcinoma).
- The chart is silent on a clinical STEM atom that the trial requires (e.g., trial wants documented Alzheimer's diagnosis; chart says nothing about Alzheimer's). **Silence on a stem clinical fact is not chart silence on a methodology qualifier.** Stems should stay FALSE on silence; do not null them.
- A real trial criterion (REQ_*_COMPONENT_*) appears in the unsat core and the patient atoms in the core directly fail that criterion.

# === MINING ARTIFACT — REQUIRES STRONG EVIDENCE ===

You may propose overrides ONLY in these specific patterns, and only when the unsat core makes the artifact unmistakable:

1. **Auxiliary-polarity bug** (rare, specific). Assertions named `AUTO_AUXILIARY_QUAL_IMPLIES_STEM_*` emit `qualifier → stem`. For negation-polarity qualifiers (`@@absent_*`, `@@without_*`, `@@off_*`, `@@no_*`, `@@not_*`), when the stem is False, the auxiliary forces the qualifier False, but semantically the absence-qualifier is vacuously True when the stem is False. ONLY apply this fix if:
   - The qualifier name explicitly matches `@@absent_*`/`@@without_*`/`@@off_*`/`@@no_*`/`@@not_*`, AND
   - The auxiliary assertion appears in the unsat core, AND
   - The stem is indeed False per the chart.

   Resolution: flip the negation qualifier to TRUE.

2. **Methodology qualifier on a chart-confirmed stem** (rare). When a stem (`patient_has_finding_of_X_now`) is TRUE per the chart, but a methodology qualifier (`@@detected_on_<modality>`, `@@diagnosed_by_<criteria>`, `@@confirmed_by_<method>`, `@@histologically_confirmed`, `@@according_to_<guideline>`) is being forced FALSE because the chart didn't name the modality, the qualifier should be NULL (chart silent on modality). ONLY apply if:
   - The stem atom is positively True per chart evidence, AND
   - The qualifier in question is a methodology qualifier (matches one of the patterns above), AND
   - The chart does not contradict the modality (no positive statement that a different modality was used).

   Resolution: set the methodology qualifier to NULL.

3. **Aggregate-vs-component definitional disagreement** (rare). A `*_count` / `*_total` / `*_score` atom appears in the core with its boolean components, and the count value disagrees with the implied count from the boolean values. ONLY apply if:
   - Both the aggregate and at least one component appear in the unsat core, AND
   - The component values clearly imply a different count than what was mined.

   Resolution: re-mine the aggregate to match the components.

4. **Temporal-state continuity auxiliary**. The compiler emits implications like `_now → _inthehistory` or `_inthehistory → _now` linking different temporal versions of the same finding. In trial context, "in history" usually means *prior to current presentation*, not "ever in life." So when:
   - The chart positively documents a CURRENT finding (`_now = True`) with no prior history (`_inthehistory = False`), AND
   - An auxiliary linking `_now → _inthehistory` (or its contrapositive) is in the unsat core,

   the auxiliary's implicit semantics ("history includes recent past including now") clash with the trial's "history" meaning prior-to-presentation.

   Resolution: set the `_inthehistory` atom to NULL, breaking the implication chain. ONLY apply if the chart positively documents the `_now` value with no prior history mentioned.

5. **Data-availability atom on a chart-silent value** (rare). Atoms named `*_is_available` / `*_is_documented` / `*_value_recorded_*_is_available` reflect whether the trial-site EHR has documented a score or value. Chart silence on such an availability atom should default to NULL (deferred to coordinator at visit), not FALSE. When this atom is in the unsat core paired with a `_DATA_AVAILABILITY` constraint and the underlying value is chart-silent, the resolution is NULL on the availability atom. ONLY apply if:
   - The atom name matches `_is_available` / `_is_documented` / `_value_recorded_*_is_available`, AND
   - A `_DATA_AVAILABILITY` constraint is present in the unsat core, AND
   - The underlying numeric / clinical value is genuinely chart-silent (not chart-False).

   Resolution: NULL the availability atom.

% Pattern 6 (stem-false-on-self-acknowledged-silence) was tested and removed:
% it gained 1 FN recovery but added 1 TN regression (net 0 TP but +1 FP).
% The LLM's distinction between "chart-silent" and "chart-contradicts" via
% evidence text was not reliable enough. Stems remain in the "do not relax"
% category.

# === DO NOT APPLY OVERRIDES IN THESE CASES ===

- A clinical stem atom (no `@@`) is False or NULL per chart silence, and the trial requires it True. → GENUINE.
- The chart describes a different finding than what the trial requires. → GENUINE (chart contradicts, not silent).
- The trial requires a documented diagnosis / lab / imaging that the chart doesn't mention. → GENUINE (the trial really requires that documentation).
- You cannot identify a specific named pattern from the three above. → `genuine_after_review`.
- The unsat core consists of a real trial criterion + patient atoms with no AUXILIARY in core. → GENUINE.

# === Inputs ===

TRIAL CRITERIA (relevant excerpt):
{trial_excerpt}

SMT PROGRAM (this side, full):
{program}

UNSAT CORE (the assertions and atoms forming the contradiction):
{unsat_core_listing}

NON-REAL STRUCTURAL ASSERTIONS IN CORE (compiler-emitted helpers, candidate suspects):
{non_real_listing}

MINED ATOM VALUES (atom = value, with chart evidence):
{atoms_listing}

PATIENT CHART (excerpt):
{chart}

# === Output ===

Output STRICT JSON, no commentary outside:
{{
  "classification": "auxiliary_polarity" | "methodology_qualifier" | "aggregate_component" | "temporal_continuity" | "data_availability" | "genuine_after_review",
  "confidence": "high" | "medium" | "low",
  "overrides": [
    {{"atom": "<atom_name>", "new_value": true | false | null,
      "rationale": "<one sentence citing the specific named pattern + why this override doesn't contradict the chart>"}}
  ],
  "explanation": "<2-3 sentences. If genuine_after_review, explain why the patient truly fails the criterion. If proposing an override, name the specific pattern and the chart evidence that justifies it.>"
}}

# === Decision rules ===
- Default classification is `genuine_after_review` with `confidence = high` and `overrides: []`.
- Only propose `overrides` if you can name ONE of the three specific patterns above AND your confidence is `high`.
- If `confidence` is `medium` or `low`, return `overrides: []` regardless of classification.
- Each override's rationale MUST cite the specific named pattern (e.g., "auxiliary_polarity: qualifier @@absent_off_corticosteroid is on a negation pattern; stem is False per chart 'neuro exam unremarkable'; flipping to TRUE makes the qualifier semantically vacuous-true").
"""


def call_llm(prompt, max_tokens=900):
    if not (ENDPOINT and KEY):
        return None
    body = json.dumps({
        'messages': [{'role':'user','content':prompt}],
        'max_tokens': max_tokens, 'temperature': 0,
        'response_format': {'type':'json_object'},
    }).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/chat/completions?api-version=2024-08-01-preview",
        data=body, headers={'api-key':KEY,'Content-Type':'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            resp = json.loads(r.read())
        return json.loads(resp['choices'][0]['message']['content'])
    except Exception as e:
        return {'_error': str(e)[:160]}


def fmt_atoms(values_rich):
    out = []
    for atom, v in (values_rich or {}).items():
        if isinstance(v, dict):
            val = v.get('value')
            ev  = (v.get('evidence','') or '')[:140].replace('\n',' ')
            asm = (v.get('assessment','') or '')[:160].replace('\n',' ')
            out.append(f'  - `{atom}` = {val}\n      evidence: {ev}\n      assessment: {asm}')
    return '\n'.join(out)


def fmt_unsat_core_listing(core, classification_info):
    lines = []
    for item in (core or []):
        if REAL_REQ_PATTERN.match(item):
            lines.append(f'  [REAL_CRITERION] {item}')
        elif item.startswith('patient_'):
            lines.append(f'  [PATIENT_ATOM]   {item}')
        else:
            lines.append(f'  [STRUCTURAL]    {item}')
    return '\n'.join(lines)


def process_one(pair, full_tid, side, mine_dir):
    side_short = 'inclusion' if side == 'inc' else 'exclusion'
    full_path = mine_dir / pair.split('__',1)[0] / f'{full_tid}__full.json'
    stats_path = mine_dir / pair.split('__',1)[0] / f'{full_tid}__{side_short}_stats.json'
    if not full_path.exists() or not stats_path.exists(): return None
    full = json.load(full_path.open())
    stats = json.load(stats_path.open())
    if stats.get('status') != 'unsat': return None
    core = stats.get('unsat_core') or []
    cls, info = classify_unsat_core(core, side_short)

    cache_key = f'{pair}__{full_tid}__{side_short}'
    cache_p = CACHE/f'{cache_key}.json'
    if cache_p.exists():
        try: return json.loads(cache_p.read_text())
        except: pass

    out = {
        'pair': pair, 'full_tid': full_tid, 'side': side_short,
        'classification_static': cls,
        'classification_info': info,
        'unsat_core': core,
    }

    if cls == 'genuine':
        out['overrides'] = []
        out['reason'] = 'all unsat-core assertions are real REQ_COMPONENT_OTHER_REQUIREMENTS; patient genuinely fails'
    else:
        # call LLM resolver
        raw = full.get(side_short, {}).get('raw') or {}
        program = '\n'.join(raw.get('smt_program_lines') or [])
        atoms = raw.get('patient_var_values_rich') or {}
        chart = raw.get('patient_contextual_text','') or ''
        if 'patient_notes' in raw and raw['patient_notes']:
            chart = chart or raw['patient_notes'][0]
        trial_inc = raw.get('trial_inclusion_criteria','') or ''
        trial_exc = raw.get('trial_exclusion_criteria','') or ''
        trial_excerpt = (trial_inc if side == 'inc' else trial_exc)[:2500]

        prompt = ARBITER_PROMPT.format(
            trial_excerpt=trial_excerpt,
            program=program[:8000],
            unsat_core_listing=fmt_unsat_core_listing(core, info),
            non_real_listing='\n'.join(f'  - `{x}`' for x in info.get('non_real_structural', [])),
            atoms_listing=fmt_atoms(atoms)[:6000],
            chart=(chart or '')[:2500],
        )
        resp = call_llm(prompt)
        if resp and not resp.get('_error'):
            out['classification_llm'] = resp.get('classification', 'unknown')
            out['confidence'] = resp.get('confidence', 'low')
            llm_overrides = resp.get('overrides') or []
            # Confidence guard: only apply overrides when LLM is HIGH confidence.
            # Low/medium confidence → trust the genuine UNSAT.
            if out['confidence'] != 'high':
                out['overrides'] = []
                out['suppressed_overrides'] = llm_overrides
            else:
                out['overrides'] = llm_overrides
            out['llm_explanation'] = resp.get('explanation','')
        else:
            out['classification_llm'] = 'llm_error'
            out['overrides'] = []
            out['confidence'] = None
            out['llm_error'] = (resp or {}).get('_error') if resp else 'no_response'

    CACHE.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    # collect all FN pairs from current asym scoring (we want to attempt to
    # rescue these) -- and run on EVERY (pair, variant, side) where the
    # corresponding mine status is unsat
    gold = json.load((ROOT/'experiments/accuracy/data/gold_5sys.json').open())['gold']

    def asym_pair_eligible(pair):
        pid, parent_nct = pair.split('__', 1)
        for d in (V10, V9):
            for f in (d/pid).glob(f'{parent_nct}*__overall.json') if (d/pid).exists() else []:
                ov = json.load(f.open())
                full_tid = f.name[:-len('__overall.json')]
                v10_o = V10/pid/f'{full_tid}__overall.json'
                v9_o  = V9/pid/f'{full_tid}__overall.json'
                inc = bool(json.load(v10_o.open()).get('inclusion_sat_like')) if v10_o.exists() else (
                      bool(json.load(v9_o.open()).get('inclusion_sat_like')) if v9_o.exists() else False)
                exc = bool(json.load(v9_o.open()).get('exclusion_sat_like')) if v9_o.exists() else (
                      bool(json.load(v10_o.open()).get('exclusion_sat_like')) if v10_o.exists() else False)
                if inc and exc:
                    return True
        return False

    # Run on ALL pairs (not just FN-side) so we can measure both rescues and
    # any new FPs from arbiter overrides on currently-correctly-rejected pairs.
    all_pairs = sorted(gold.keys())
    if args.limit: all_pairs = all_pairs[:args.limit]
    print(f'pairs to arbitrate (all UNSAT cases): {len(all_pairs)}')

    tasks = []
    for pair in all_pairs:
        pid, parent_nct = pair.split('__', 1)
        for d, side, mine_dir in [(V10, 'inc', V10), (V9, 'exc', V9)]:
            pdir = d/pid
            if not pdir.exists(): continue
            for f in pdir.glob(f'{parent_nct}*__{side}lusion_stats.json' if False else f'{parent_nct}*__{"inclusion" if side=="inc" else "exclusion"}_stats.json'):
                tid = f.name.replace(f'__{"inclusion" if side=="inc" else "exclusion"}_stats.json','')
                try: st = json.load(f.open())
                except: continue
                if st.get('status') == 'unsat':
                    tasks.append((pair, tid, side, mine_dir))

    print(f'UNSAT tasks: {len(tasks)}')

    n_done = 0; static_genuine = 0; static_mining = 0; resolved = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_one, p, t, s, d) for (p,t,s,d) in tasks]
        for f in as_completed(futs):
            try: r = f.result()
            except Exception as e:
                print(f'  err: {e}'); continue
            n_done += 1
            if not r: continue
            if r['classification_static'] == 'genuine': static_genuine += 1
            else:
                static_mining += 1
                if r.get('overrides'): resolved += 1
            if n_done % 10 == 0:
                print(f'  [{n_done}/{len(tasks)}] genuine={static_genuine} mining={static_mining} resolved={resolved}', flush=True)

    print(f'\nDone. genuine={static_genuine} mining-self-contradiction={static_mining} (resolved={resolved})')
    print(f'Cache: {CACHE}')


if __name__ == '__main__':
    main()
