#!/usr/bin/env python3
"""Verbalize AEGIS's symbolic chain into plain-English rationale (using existing
verbalize_smt_rationale.prompt). Then we can fair-compare with V5's NL rationale
in the adjudicator (both NL-form, no concrete-symbolic vs free-text framing bias).
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments/99_counterfactual_lm'))
try:
    from run_better_nl_full import collect_pairs
except ModuleNotFoundError as _e:  # pragma: no cover
    raise ModuleNotFoundError(
        "run_better_nl_full is not part of this repository -- this batch driver "
        "depends on an internal module that was not released. The "
        "supported entry points are the `verdict` command and the "
        "`verdict` Python package; see README."
    ) from _e

VERBALIZE_PROMPT_PATH = ROOT / "verbalizer/prompts/verbalize_smt_rationale.prompt"


def yes(s): return (s or '').strip().lower() in ('eligible','yes','forward','true','1')


_PRECOMPUTED_CORES = {}
_SIGIR_CORPUS = None
_HEADLINE_LABELS = {}
def _load_headline_labels():
    global _HEADLINE_LABELS
    if _HEADLINE_LABELS: return
    p = pathlib.Path(os.environ.get("VERDICT_ROOT",
        pathlib.Path(__file__).resolve().parents[1])) / "overnight/aegis_strict_inc.jsonl"
    if p.exists():
        for ln in p.open():
            r = json.loads(ln); _HEADLINE_LABELS[r['pair']] = r.get('eligibility')
def _get_corpus():
    global _SIGIR_CORPUS
    if _SIGIR_CORPUS is None:
        _SIGIR_CORPUS = {}
        p = pathlib.Path(os.environ.get("SIGIR_CORPUS",
        pathlib.Path(__file__).resolve().parents[1].parent
        / "TrialGPT-SMT/dataset/clinical_trial/sigir/corpus.jsonl"))
        if p.exists():
            for ln in p.open():
                r = json.loads(ln); _SIGIR_CORPUS[r['_id']] = r
    return _SIGIR_CORPUS


def build_inputs(pair, full):
    """Build inputs for the verbalize prompt from cmsrc full.json.

    Uses strict-inc policy (matching aegis_strict_inc.jsonl headline) and
    computes strict-inc UNSAT cores via Z3 so the rationale is faithful to
    the headline verdict.
    """
    pid, nct = pair.split('__', 1)
    inc_d = full.get('inclusion') or {}
    exc_d = full.get('exclusion') or {}
    inc_raw = inc_d.get('raw') or {}
    exc_raw = exc_d.get('raw') or {}

    note = inc_raw.get('patient_contextual_text') or ''
    if not note:
        pat = inc_raw.get('patient', {})
        note = (pat.get('text') or '')[:3000]

    # Use canonical SIGIR corpus text (full, no truncation) so the verbalizer's trial text matches what V5/judges saw.
    corpus = _get_corpus()
    parent_id = re.sub(r'(?<=NCT\d{8})[a-z]+$', '', nct)
    trial_text = ''
    if parent_id in corpus:
        trial_text = corpus[parent_id].get('text','') or ''
    if not trial_text:
        trial = (inc_raw.get('trial_inclusion_criteria') or '') + '\n\nEXCLUSION:\n' + (exc_raw.get('trial_exclusion_criteria') or '')
    else:
        trial = trial_text

    # Use precomputed strict-inc cores (Z3 is not thread-safe → can't run inside ThreadPoolExecutor).
    pc = _PRECOMPUTED_CORES.get(pair, {}) if _PRECOMPUTED_CORES else {}
    inc_v = pc.get('inc_v', inc_d.get('sat_like'))
    exc_v = pc.get('exc_v', exc_d.get('sat_like'))
    inc_atoms = (pc.get('inc_core_atoms') or inc_d.get('summary',{}).get('unsat_core', []))[:30]
    exc_atoms = (pc.get('exc_core_atoms') or exc_d.get('summary',{}).get('unsat_core', []))[:30]

    # Use the headline AEGIS verdict directly (read from aegis_strict_inc.jsonl).
    # This guarantees the verbalized rationale's label matches the system we report.
    label = _HEADLINE_LABELS.get(pair) or ('eligible' if pc.get('aggregate_eligible') else 'ineligible')

    artifacts = {
        'policy': 'strict-inclusion (silence on inclusion fails; silence on exclusion is benign)',
        'inclusion_status': 'SAT' if inc_v is True else ('UNSAT' if inc_v is False else 'unknown'),
        'inclusion_unsat_core': inc_atoms,  # atoms whose patient values contradict the inclusion → patient FAILS inclusion
        'exclusion_status': 'SAT' if exc_v is True else ('UNSAT' if exc_v is False else 'unknown'),
        'exclusion_unsat_core': exc_atoms,  # atoms whose patient values trigger an exclusion → patient MEETS exclusion
        'final_label': label,
    }

    def fmt_pv(pv):
        out = []
        for atom, m in list(pv.items())[:50]:
            v = m.get('value')
            if v is None: continue
            ev = (m.get('evidence') or '')[:300]
            ra = (m.get('assessment') or '')[:200]
            out.append(f"  {atom} = {v}\n    evidence (chart): \"{ev}\"\n    why: {ra}")
        return '\n'.join(out)

    inc_pv_str = fmt_pv(inc_raw.get('patient_var_values_rich') or {})
    exc_pv_str = fmt_pv(exc_raw.get('patient_var_values_rich') or {})

    return {
        'note': note[:3000], 'trial': trial[:5000],
        'label': label,
        'artifacts': json.dumps(artifacts, indent=2)[:2000],
        'inc_pv': inc_pv_str[:5000],
        'exc_pv': exc_pv_str[:5000],
    }


def call(engine, prompt, max_tokens=3000):
    out = engine([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=max_tokens)
    text = out[0] if isinstance(out, list) and out else out
    text = text if isinstance(text, str) else str(text)
    jm = re.search(r"\{[\s\S]*\}", text)
    if not jm: return None
    try: return json.loads(jm.group(0))
    except: return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--disagreements-only', action='store_true', help='Only verbalize AEGIS for AEGIS-V5 disagreements')
    args = ap.parse_args()

    OUT = ROOT / 'overnight' / 'aegis_verbalized.jsonl'
    cache = {}
    if OUT.exists():
        for line in OUT.open():
            try: o = json.loads(line); cache[o['pair']] = o
            except: pass

    pairs, _ = collect_pairs()
    if args.limit: pairs = pairs[:args.limit]

    if args.disagreements_only:
        # Filter to disagreements
        v5 = {}
        for line in open(ROOT / 'overnight/v5_verbose.jsonl'):
            o = json.loads(line); v5[o['pair']] = (o.get('eligibility') == 'eligible')
        aegis = {}
        for p in (ROOT / 'experiments/53_v2_full/judges_clinician_v2').glob('*.json'):
            o = json.loads(p.read_text()); pair = o.get('pair')
            if pair: aegis[pair] = yes(o['per_system'].get('aegis',{}).get('decision',''))
        pairs = [p for p in pairs if p in aegis and p in v5 and aegis[p] != v5[p]]
        print(f"Disagreement-only: {len(pairs)} pairs", flush=True)

    _load_headline_labels()
    todo = [p for p in pairs if p not in cache]
    print(f"[verbalize-aegis] cached={len(cache)} todo={len(todo)} headline labels={len(_HEADLINE_LABELS)}", flush=True)
    if not todo: return

    # Precompute strict-inc UNSAT cores SEQUENTIALLY across ALL cohort variants
    # and aggregate per pair (any variant eligible → eligible). Pick the variant
    # used for verbalization based on the aggregate verdict:
    #   if eligible: pick an eligible variant (rationale will explain why this variant passed)
    #   if ineligible: pick the variant with the strongest UNSAT cores so the rationale is informative
    # Z3 is not thread-safe → keep this loop sequential.
    print(f"[verbalize-aegis] precomputing UNSAT cores for {len(todo)} pairs (multi-variant)...", flush=True)
    from overnight.extract_unsat_cores import solve_with_core
    t0 = time.time()
    for i, pair in enumerate(todo, 1):
        pid, nct = pair.split('__', 1)
        pdir = ROOT / 'experiments/53_v2_full/cmsrc_out' / pid
        files = sorted(pdir.glob(f'{nct}*__full.json'))
        if not files: continue
        variants = []
        for vf in files:
            try:
                full = json.loads(vf.read_text())
                inc_raw = (full.get('inclusion') or {}).get('raw') or {}
                exc_raw = (full.get('exclusion') or {}).get('raw') or {}
                inc_v, inc_core, inc_lmap = solve_with_core(
                    inc_raw.get('smt_program_lines') or [],
                    inc_raw.get('patient_var_values_rich') or {}, side='inclusion')
                exc_v, exc_core, exc_lmap = solve_with_core(
                    exc_raw.get('smt_program_lines') or [],
                    exc_raw.get('patient_var_values_rich') or {}, side='exclusion')
                inc_atoms = [inc_lmap[l][1] for l in (inc_core or []) if l in inc_lmap]
                exc_atoms = [exc_lmap[l][1] for l in (exc_core or []) if l in exc_lmap]
                eligible = (inc_v is True) and (exc_v is not False)
                variants.append({
                    'file': vf, 'inc_v': inc_v, 'exc_v': exc_v,
                    'inc_atoms': inc_atoms, 'exc_atoms': exc_atoms,
                    'eligible': eligible,
                })
            except Exception:
                continue
        if not variants: continue
        any_eligible = any(v['eligible'] for v in variants)
        if any_eligible:
            chosen = next(v for v in variants if v['eligible'])
        else:
            # Pick variant with most informative cores (most core atoms)
            chosen = max(variants, key=lambda v: len(v['inc_atoms']) + len(v['exc_atoms']))
        _PRECOMPUTED_CORES[pair] = {
            'file': chosen['file'],
            'inc_v': chosen['inc_v'], 'exc_v': chosen['exc_v'],
            'inc_core_atoms': chosen['inc_atoms'],
            'exc_core_atoms': chosen['exc_atoms'],
            'aggregate_eligible': any_eligible,
            'n_variants': len(variants),
        }
        if i % 50 == 0:
            print(f"  cores [{i}/{len(todo)}] ({time.time()-t0:.0f}s)", flush=True)
    print(f"[verbalize-aegis] cores done in {time.time()-t0:.0f}s", flush=True)

    template = VERBALIZE_PROMPT_PATH.read_text()

    from smt_core.inference_engine import AzureInferenceEngine
    endpoint = os.environ['OPENAI_ENDPOINT']
    engine = AzureInferenceEngine(endpoint=endpoint, model_name='gpt-4.1').run

    def proc(pair):
        last = "no_response"
        # Use the variant chosen during precompute (matches aggregate AEGIS verdict)
        pc = _PRECOMPUTED_CORES.get(pair)
        if pc and pc.get('file'):
            full = json.loads(pc['file'].read_text())
        else:
            pid, nct = pair.split('__', 1)
            pdir = ROOT / 'experiments/53_v2_full/cmsrc_out' / pid
            files = sorted(pdir.glob(f'{nct}*__full.json'))
            if not files: return pair, {'pair': pair, 'error': 'no_full_json'}
            full = json.loads(files[0].read_text())
        inputs = build_inputs(pair, full)
        prompt = (template
                  .replace('#PATIENT_NOTE#', inputs['note'])
                  .replace('#TRIAL_ELIGIBILITY_TEXT#', inputs['trial'])
                  .replace('#SYSTEM_DECISION_LABEL#', inputs['label'])
                  .replace('#SYSTEM_DECISION_ARTIFACTS#', inputs['artifacts'])
                  .replace('#INCLUSION_EXTRACTED_VARIABLE_VALUES#', inputs['inc_pv'])
                  .replace('#EXCLUSION_EXTRACTED_VARIABLE_VALUES#', inputs['exc_pv']))
        for attempt in range(3):
            try:
                res = call(engine, prompt)
                if res:
                    return pair, {
                        'pair': pair,
                        'aegis_label': inputs['label'],
                        'verbalized_rationale': res.get('rationale', '')[:1500],
                        'key_points': (res.get('key_points') or [])[:8],
                    }
            except Exception as e:
                last = e; time.sleep(1+attempt)
        return pair, {'pair': pair, 'error': str(last)[:120]}

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(proc, p) for p in todo]
        with OUT.open('a') as fout:
            for f in as_completed(futs):
                pair, rec = f.result()
                cache[pair] = rec
                fout.write(json.dumps(rec) + '\n'); fout.flush()
                done += 1
                if done % 30 == 0 or done == len(todo): print(f"  [{done}/{len(todo)}]", flush=True)
    print(f'[verbalize-aegis] FINAL: {len(cache)} verbalizations', flush=True)


if __name__ == '__main__':
    main()
